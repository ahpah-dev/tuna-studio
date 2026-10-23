"""Resident offline inference worker. NDJSON commands in, streamed events out."""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import sys
import threading
import time

from chat_service import validate_message_request
from tuna_core import prepare_inference_messages, reserved_control_tokens
from inference_loading import checkpoint_size, choose_precision
from chat_options import ReasoningSplitter, response_budget, reasoning_supported, finish_reason

PRINT_LOCK = threading.Lock()


def emit(kind, **fields):
    with PRINT_LOCK:
        print(json.dumps({"type":kind, **fields},ensure_ascii=False,allow_nan=False),flush=True)


def prepare_messages(messages, manifest=None):
    clean = validate_message_request({"messages":messages})["messages"]
    return prepare_inference_messages(clean, manifest) if manifest else clean


def end_token_ids(model, tokenizer):
    value = model.generation_config.eos_token_id
    result = set(value if isinstance(value,list) else [value] if value is not None else [])
    if tokenizer.eos_token_id is not None:
        result.add(tokenizer.eos_token_id)
    vocab = tokenizer.get_vocab()
    for token in ("<|im_end|>","<|eot_id|>","<|end|>","<end_of_turn>"):
        if token in vocab:
            result.add(vocab[token])
    return sorted(result)


def load_model(args):
    os.environ.update(HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",HF_HUB_DISABLE_TELEMETRY="1",TOKENIZERS_PARALLELISM="false")
    emit("progress",message="Opening your saved model and policy…")
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM,AutoTokenizer,BitsAndBytesConfig
    run = args.run_dir.resolve()
    config = json.loads((run/"config.json").read_text(encoding="utf-8"))
    manifest = None
    if config.get("method") == "tuna":
        manifest = json.loads((run/"policy.json").read_text(encoding="utf-8"))
        prepare_inference_messages([{"role":"user","content":"validate"}],manifest)
    if args.source == "gguf":
        from gguf_import import import_gguf
        export = json.loads((run/"gguf"/"export_manifest.json").read_text(encoding="utf-8"))
        source = (run/"gguf"/export["filename"]).resolve()
        if source.parent != (run/"gguf").resolve() or source.suffix.lower() != ".gguf":
            raise ValueError("The saved GGUF path is invalid.")
        base_path,_ = import_gguf(source,args.cache_dir,args.work_dir,progress=lambda message:emit("progress",message=message))
        tokenizer_path, adapter_path = base_path, None
    else:
        summary = json.loads((run/"training_summary.json").read_text(encoding="utf-8"))
        base_path = summary.get("model_path",config["model_path"])
        tokenizer_path = adapter_path = run/"adapter"
    cuda = torch.cuda.is_available()
    if config.get("method") in ("tuna","qlora") and not cuda:
        raise RuntimeError("This fine-tune needs CUDA. Check the local engine in Settings.")
    dtype = (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16) if cuda else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path,local_files_only=True,trust_remote_code=False,use_fast=True)
    if not tokenizer.chat_template:
        raise ValueError("This model does not have a saved chat template.")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    free,total = torch.cuda.mem_get_info() if cuda else (0,0)
    precision = choose_precision(checkpoint_size(base_path),free,total,cuda=cuda,
        bf16=cuda and torch.cuda.is_bf16_supported(),preference=os.environ.get("TUNA_CHAT_PRECISION","auto"))
    def open_weights(selected):
        kwargs = {"local_files_only":True,"trust_remote_code":False,"dtype":dtype,"attn_implementation":"sdpa"}
        if cuda:
            kwargs["device_map"] = {"":0}
        if selected == "nf4":
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True,
                bnb_4bit_quant_type="nf4",bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=dtype)
        loaded = AutoModelForCausalLM.from_pretrained(base_path,**kwargs)
        if adapter_path:
            loaded = PeftModel.from_pretrained(loaded,adapter_path,local_files_only=True)
            if selected != "nf4":
                emit("progress",message="Merging your trained adapter in memory for faster replies…")
                loaded = loaded.merge_and_unload(safe_merge=True)
        return loaded
    emit("progress",message="Loading model weights on your GPU…" if cuda else "Loading model weights on your CPU…")
    try:
        try:
            model = open_weights(precision)
        except torch.cuda.OutOfMemoryError:
            if precision == "nf4" or not cuda:
                raise
            model = None
        if model is None:
            gc.collect()
            torch.cuda.empty_cache()
            precision = "nf4"
            emit("progress",message="Using the lower-memory 4-bit path to fit your GPU…")
            model = open_weights(precision)
    except OSError as exc:
        raise RuntimeError("The base model files for this run are missing locally. Choose its GGUF export, or restore its base model files. Chat does not download models. " + str(exc)[:350]) from exc
    model.eval()
    model.config.use_cache = True
    from inference_optimizations import install_optimizations
    optimizations = install_optimizations(model)
    context = min(8192,int(getattr(model.config,"max_position_embeddings",8192) or 8192))
    compiled = False
    # Compile optional kernels and initialize CUDA before enabling the composer,
    # rather than making the user's first reply pay that one-time cost.
    if cuda:
        emit("progress",message="Preparing fast inference kernels… The first load can take a minute.")
        cache_planner = getattr(model,"_prepare_static_cache",None)
        cache_code = getattr(getattr(cache_planner,"__func__",cache_planner),"__code__",None)
        stable_static_cache = cache_code is not None and "_previous_max_cache_length" in cache_code.co_names
        compile_candidate = (stable_static_cache and precision in ("bf16","fp16")
            and "Qwen3.5 fused linear attention" in optimizations
            and os.environ.get("TUNA_CHAT_COMPILE","1") != "0")
        if compile_candidate:
            from transformers import CompileConfig
            emit("progress",message="Compiling the fast GPU decoder… First-time preparation can take several minutes.")
            model._previous_max_cache_length = context
            model.generation_config.cache_implementation = "static"
            model.generation_config.compile_config = CompileConfig(fullgraph=True,mode="reduce-overhead")
        def disable_compilation():
            model.generation_config.cache_implementation = None
            model.generation_config.compile_config = None
            model.generation_config.disable_compile = True
            model.__dict__.pop("_compiled_call",None)
            torch._dynamo.reset()
            gc.collect()
            torch.cuda.empty_cache()
        def warmup():
            prepared = prepare_messages([{"role":"user","content":"Hello."}],manifest)
            prompt = tokenizer.apply_chat_template(prepared,tokenize=False,add_generation_prompt=True,enable_thinking=False)
            inputs = tokenizer(prompt,return_tensors="pt",add_special_tokens=False,return_token_type_ids=False).to(model.device)
            if inputs["input_ids"].shape[1]+3 > context:
                raise ValueError("The saved policy exceeds this model's chat context budget.")
            with torch.inference_mode():
                model.generate(**inputs,max_new_tokens=3,min_new_tokens=3,do_sample=False,use_cache=True,
                    pad_token_id=tokenizer.pad_token_id,eos_token_id=end_token_ids(model,tokenizer) or None)
            torch.cuda.synchronize()
        compatible_fallback,reference_fallback = False,False
        try:
            warmup()
            if compile_candidate:
                warmup()
                compiled = hasattr(model,"_compiled_call")
        except torch.cuda.OutOfMemoryError:
            if not compile_candidate:
                raise
            compatible_fallback = True
        except Exception:
            if compile_candidate:
                compatible_fallback = True
            elif "Qwen3.5 fused linear attention" in optimizations:
                reference_fallback = True
            else:
                raise
        # End the exception scope before collecting failed graph/cache frames;
        # otherwise a traceback can retain their GPU memory during the retry.
        if compatible_fallback:
            disable_compilation()
            emit("progress",message="Using compatible fast inference without compiled decode…")
            try:
                warmup()
            except torch.cuda.OutOfMemoryError:
                raise
            except Exception:
                if "Qwen3.5 fused linear attention" not in optimizations:
                    raise
                reference_fallback = True
        if reference_fallback:
            gc.collect()
            torch.cuda.empty_cache()
            emit("progress",message="Optional kernels could not initialize. Using compatible local inference…")
            optimizations = install_optimizations(model,allow_fused=False)
            warmup()
        if compiled:
            optimizations.append("Compiled static-cache decode")
    emit("ready",device=torch.cuda.get_device_name(0) if cuda else "CPU",quantized=precision=="nf4",
         policy_bound=manifest is not None,context_tokens=context,engine="transformers",precision=precision,
         adapter_merged=bool(adapter_path and precision!="nf4"),optimizations=optimizations,
         compiled=compiled,
         reasoning_supported=reasoning_supported(tokenizer,prepare_messages([{"role":"user","content":"Hello."}],manifest)),
         allocated_vram_mb=torch.cuda.memory_allocated()/1048576 if cuda else None)
    return model,tokenizer,manifest,context


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir",type=Path,required=True)
    parser.add_argument("--source",choices=("adapter","gguf"),required=True)
    parser.add_argument("--cache-dir",type=Path,required=True)
    parser.add_argument("--work-dir",type=Path,required=True)
    args = parser.parse_args()
    try:
        model,tokenizer,manifest,context = load_model(args)
    except Exception as exc:
        emit("fatal_error",message=str(exc))
        return 1
    import torch
    from transformers import TextStreamer,StoppingCriteria,StoppingCriteriaList
    cancel,active = threading.Event(),threading.Event()
    generation_thread = None

    class CancelCriteria(StoppingCriteria):
        def __call__(self,input_ids,scores,**kwargs):
            return torch.full((input_ids.shape[0],),cancel.is_set(),device=input_ids.device,dtype=torch.bool)

    def generate(command):
        identifier = command["request_id"]
        chunks,thoughts,ended = [],[],False
        blocked_controls = set()
        splitter = ReasoningSplitter()
        first_token,first_text,start = None,None,None
        class Stream(TextStreamer):
            def put(self,value):
                nonlocal first_token
                if not (self.skip_prompt and self.next_tokens_are_prompt) and first_token is None:
                    first_token = time.perf_counter()-start
                if blocked_controls and not self.next_tokens_are_prompt:
                    ids = value.reshape(-1).tolist()
                    kept = [token for token in ids if token not in blocked_controls]
                    if not kept:
                        return
                    if len(kept) != len(ids):
                        value = torch.tensor(kept,dtype=value.dtype)
                super().put(value)
            def on_finalized_text(self,text,stream_end=False):
                nonlocal ended,first_text
                ended = ended or stream_end
                if text:
                    if first_text is None:
                        first_text = time.perf_counter()-start
                    self.route(splitter.feed(text))
                if stream_end:
                    self.route(splitter.finish())
                emit("reply_state",request_id=identifier,reasoning_active=splitter.in_reasoning)
            def route(self,pieces):
                for channel,text in pieces:
                    (thoughts if channel=="reasoning" else chunks).append(text)
                    emit("delta",request_id=identifier,channel=channel,text=text)
        streamer = Stream(tokenizer,skip_prompt=True,skip_special_tokens=command.get("reasoning") is not True)
        final = {}
        try:
            request = validate_message_request(command)
            prepared = prepare_messages(request["messages"],manifest)
            if request["reasoning"] and not reasoning_supported(tokenizer,prepared):
                raise ValueError("This model does not have a native reasoning switch. Turn Reasoning off or choose a supported model.")
            if request["reasoning"]:
                vocab = tokenizer.get_vocab()
                blocked_controls = {vocab[token] for token in reserved_control_tokens(tokenizer)
                    if token in vocab and token not in ("<think>","</think>")}
            if manifest:
                controls = reserved_control_tokens(tokenizer)
                for message in prepared[1:]:
                    if any(token in message["content"] for token in controls):
                        raise ValueError("Message contains reserved model control tokens. The saved Tuna policy is kept separate from chat text.")
            prompt = tokenizer.apply_chat_template(prepared,tokenize=False,add_generation_prompt=True,enable_thinking=request["reasoning"])
            splitter = ReasoningSplitter(request["reasoning"],prompt)
            encoded = tokenizer(prompt,return_tensors="pt",add_special_tokens=False,return_token_type_ids=False).to(model.device)
            prompt_tokens = encoded["input_ids"].shape[1]
            budget = response_budget(request["max_new_tokens"],prompt_tokens,context)
            emit("reply_state",request_id=identifier,reasoning_active=splitter.in_reasoning,output_token_limit=budget)
            settings = {"max_new_tokens":budget,"do_sample":request["temperature"]>0,
                "streamer":streamer,"stopping_criteria":StoppingCriteriaList([CancelCriteria()]),
                "pad_token_id":tokenizer.pad_token_id,"use_cache":True}
            eos = end_token_ids(model,tokenizer)
            if eos:
                settings["eos_token_id"] = eos
            if settings["do_sample"]:
                settings.update(temperature=request["temperature"],top_p=0.9)
            if model.device.type == "cuda":
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            with torch.inference_mode():
                output = model.generate(**encoded,**settings)
            if model.device.type == "cuda":
                torch.cuda.synchronize()
            duration = time.perf_counter() - start
            count = int(output.shape[1] - prompt_tokens)
            final = {"type":"reply_complete","request_id":identifier,"text":"".join(chunks),"reasoning":"".join(thoughts),
                "finish_reason":finish_reason(cancel.is_set(),count,budget,request["max_new_tokens"] is None,int(output[0,-1]),eos),
                "usage":{"input_tokens":prompt_tokens,"output_tokens":count,"elapsed_seconds":duration,
                         "unlimited":request["max_new_tokens"] is None,"output_token_limit":budget,
                         "reasoning_enabled":request["reasoning"],"reasoning_incomplete":request["reasoning"] and splitter.in_reasoning,
                         "tokens_per_second":count/max(duration,0.001),
                         "time_to_first_token_seconds":first_token,"time_to_first_text_seconds":first_text,
                         "decode_tokens_per_second":(count-1)/max(duration-first_token,0.001) if first_token is not None and count>1 else None,
                         "peak_allocated_vram_mb":torch.cuda.max_memory_allocated()/1048576 if model.device.type=="cuda" else None}}
            del output,encoded
        except Exception as exc:
            if not ended:
                streamer.end()
            error = "Not enough GPU memory. Try a shorter conversation or unload other models." if isinstance(exc,torch.cuda.OutOfMemoryError) else str(exc)
            final = {"type":"reply_error","request_id":identifier,"text":"".join(chunks),"reasoning":"".join(thoughts),"error":error,"finish_reason":"error"}
        finally:
            active.clear()
            kind = final.pop("type")
            emit(kind,**final)

    try:
        for line in sys.stdin:
            command = json.loads(line)
            if command.get("type") == "stop":
                cancel.set()
            elif command.get("type") == "generate":
                if active.is_set():
                    emit("reply_error",request_id=command["request_id"],error="The model is already replying.",finish_reason="error")
                    continue
                if generation_thread:
                    generation_thread.join(timeout=1)
                cancel.clear()
                active.set()
                generation_thread = threading.Thread(target=generate,args=(command,),daemon=True)
                generation_thread.start()
    finally:
        cancel.set()
        if generation_thread:
            generation_thread.join(timeout=3)
    return 0


if __name__ == "__main__":
    from app_runtime import use_app_environment
    use_app_environment(__file__)
    raise SystemExit(main())
