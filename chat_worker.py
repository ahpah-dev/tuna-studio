"""Resident offline inference worker. NDJSON commands in, streamed events out."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import threading
import time

from chat_service import validate_message_request
from tuna_core import prepare_inference_messages, reserved_control_tokens

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
    kwargs = {"local_files_only":True,"trust_remote_code":False,"dtype":dtype,"attn_implementation":"sdpa"}
    if cuda:
        kwargs.update(device_map={"":0},quantization_config=BitsAndBytesConfig(load_in_4bit=True,
            bnb_4bit_quant_type="nf4",bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=dtype))
    emit("progress",message="Loading model weights on your GPU…" if cuda else "Loading model weights on your CPU…")
    try:
        model = AutoModelForCausalLM.from_pretrained(base_path,**kwargs)
    except OSError as exc:
        raise RuntimeError("The base model files for this run are missing locally. Choose its GGUF export, or restore its base model files. Chat does not download models. " + str(exc)[:350]) from exc
    if adapter_path:
        model = PeftModel.from_pretrained(model,adapter_path,local_files_only=True)
    model.eval()
    model.config.use_cache = True
    context = min(8192,int(getattr(model.config,"max_position_embeddings",8192) or 8192))
    emit("ready",device=torch.cuda.get_device_name(0) if cuda else "CPU",quantized=cuda,
         policy_bound=manifest is not None,context_tokens=context)
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
        chunks,ended = [],False
        class Stream(TextStreamer):
            def on_finalized_text(self,text,stream_end=False):
                nonlocal ended
                ended = ended or stream_end
                if text:
                    chunks.append(text)
                    emit("delta",request_id=identifier,text=text)
        streamer = Stream(tokenizer,skip_prompt=True,skip_special_tokens=True)
        final = {}
        try:
            request = validate_message_request(command)
            prepared = prepare_messages(request["messages"],manifest)
            if manifest:
                controls = reserved_control_tokens(tokenizer)
                for message in prepared[1:]:
                    if any(token in message["content"] for token in controls):
                        raise ValueError("Message contains reserved model control tokens. The saved Tuna policy is kept separate from chat text.")
            prompt = tokenizer.apply_chat_template(prepared,tokenize=False,add_generation_prompt=True,enable_thinking=False)
            encoded = tokenizer(prompt,return_tensors="pt",add_special_tokens=False,return_token_type_ids=False).to(model.device)
            prompt_tokens = encoded["input_ids"].shape[1]
            if prompt_tokens + request["max_new_tokens"] > context:
                raise ValueError(f"This conversation exceeds the {context:,}-token chat budget. Start a new chat or shorten your message; the saved policy will not be truncated.")
            settings = {"max_new_tokens":request["max_new_tokens"],"do_sample":request["temperature"]>0,
                "streamer":streamer,"stopping_criteria":StoppingCriteriaList([CancelCriteria()]),
                "pad_token_id":tokenizer.pad_token_id,"use_cache":True}
            eos = end_token_ids(model,tokenizer)
            if eos:
                settings["eos_token_id"] = eos
            if settings["do_sample"]:
                settings.update(temperature=request["temperature"],top_p=0.9)
            start = time.perf_counter()
            with torch.inference_mode():
                output = model.generate(**encoded,**settings)
            duration = time.perf_counter() - start
            count = int(output.shape[1] - prompt_tokens)
            final = {"type":"reply_complete","request_id":identifier,"text":"".join(chunks),
                "finish_reason":"cancelled" if cancel.is_set() else "length" if count>=request["max_new_tokens"] else "stop",
                "usage":{"input_tokens":prompt_tokens,"output_tokens":count,"elapsed_seconds":duration,
                         "tokens_per_second":count/max(duration,0.001)}}
            del output,encoded
        except Exception as exc:
            if not ended:
                streamer.end()
            error = "Not enough GPU memory. Try a shorter conversation or unload other models." if isinstance(exc,torch.cuda.OutOfMemoryError) else str(exc)
            final = {"type":"reply_error","request_id":identifier,"text":"".join(chunks),"error":error,"finish_reason":"error"}
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
