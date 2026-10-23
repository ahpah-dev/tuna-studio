#!/usr/bin/env python3
"""Tuna Studio's loopback-only, standard-library application server.

Run: python server.py --port 8765
Training is delegated to trainer.py in a separate process. The server never
simulates a training run, fetches a model, or installs an ML dependency.
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import math
import mimetypes
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote, urlsplit
import uuid
import zipfile

from gguf_import import local_gguf_path, clean_import_work
from chat_service import ChatError, ChatSession

VERSION = "0.1.0"
MAX_BODY = 25 * 1024 * 1024
IDENTIFIER = re.compile(r"^[a-f0-9]{32}$")
BASE_DIR = Path(__file__).resolve().parent
DEPENDENCIES = ("torch", "transformers", "peft", "accelerate", "bitsandbytes")


def now():
    return datetime.now(timezone.utc).isoformat()


class APIError(Exception):
    def __init__(self, message, status=400):
        super().__init__(message)
        self.status = status


def atomic_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def normalize_jsonl(content):
    """Validate and normalize complete chat examples, with useful line errors."""
    if not isinstance(content, str) or not content.strip():
        raise APIError("The JSONL file is empty. Add at least one training example.")
    if len(content.encode("utf-8")) > MAX_BODY:
        raise APIError("Dataset exceeds the 25 MB upload limit.", 413)
    records = []
    for line_number, line in enumerate(content.lstrip("\ufeff").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except (ValueError, RecursionError):
            raise APIError(f"Line {line_number}: invalid JSON. Each line must be a complete JSON object.")
        if not isinstance(record, dict):
            raise APIError(f"Line {line_number}: expected an object with messages or prompt/completion.")
        if "messages" in record:
            messages = record["messages"]
        elif "prompt" in record and "completion" in record:
            messages = [{"role": "user", "content": record["prompt"]},
                        {"role": "assistant", "content": record["completion"]}]
        else:
            raise APIError(f"Line {line_number}: use messages or prompt/completion fields.")
        if not isinstance(messages, list) or not 2 <= len(messages) <= 1000:
            raise APIError(f"Line {line_number}: messages must contain 2 to 1,000 chat messages.")
        normalized = []
        for message in messages:
            if not isinstance(message, dict) or message.get("role") not in ("system", "user", "assistant"):
                raise APIError(f"Line {line_number}: each role must be system, user, or assistant.")
            value = message.get("content")
            if not isinstance(value, str) or not value.strip():
                raise APIError(f"Line {line_number}: every message must have nonempty text content.")
            normalized.append({"role": message["role"], "content": value})
        roles = {message["role"] for message in normalized}
        if not {"user", "assistant"}.issubset(roles):
            raise APIError(f"Line {line_number}: include at least one user and one assistant message.")
        if any(message["role"] == "system" for message in normalized[1:]):
            raise APIError(f"Line {line_number}: only one leading system message is supported.")
        conversation = normalized[1:] if normalized[0]["role"] == "system" else normalized
        if conversation[-1]["role"] != "assistant":
            raise APIError(f"Line {line_number}: each conversation must end with an assistant reply.")
        if any(message["role"] != ("user" if index % 2 == 0 else "assistant") for index, message in enumerate(conversation)):
            raise APIError(f"Line {line_number}: conversation roles must alternate user, assistant.")
        records.append({"messages": normalized})
        if len(records) > 100000:
            raise APIError("A dataset may contain at most 100,000 examples.")
    if not records:
        raise APIError("The JSONL file contains no training examples.")
    return records


def finite_number(config, key, default, minimum, maximum, integer=False):
    value = config.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise APIError(f"{key} must be a finite number.")
    if integer and int(value) != value:
        raise APIError(f"{key} must be an integer.")
    if value < minimum or value > maximum:
        raise APIError(f"{key} must be between {minimum} and {maximum}.")
    return int(value) if integer else float(value)


PROBE_CODE = r'''
import importlib, json, subprocess
result = {"dependencies": {}, "dependency_errors": {}, "device": {"name": "CPU", "vram_total_gb": 0, "vram_used_gb": 0, "cuda": False}}
try:
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=8, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if gpu.returncode == 0 and gpu.stdout.strip():
        name, total, used = gpu.stdout.strip().splitlines()[0].rsplit(",", 2)
        result["device"].update(name=name.strip(), vram_total_gb=round(float(total)/1024, 2), vram_used_gb=round(float(used)/1024, 2), detected_gpu=True)
except Exception:
    pass
for name in ("torch", "transformers", "peft", "accelerate", "bitsandbytes"):
    try:
        module = importlib.import_module(name)
        result["dependencies"][name] = True
        if name == "torch":
            result["torch_version"] = module.__version__
            result["cuda_runtime"] = module.version.cuda
            available = module.cuda.is_available()
            result["device"]["cuda"] = available
            if available:
                # Driver detection alone does not prove that this wheel supports
                # the GPU architecture. Execute and synchronize a real kernel.
                sample = module.ones((16, 16), device="cuda")
                product = sample @ sample
                module.cuda.synchronize()
                assert product[0, 0].item() == 16
                result["device"]["kernel_verified"] = True
                del sample, product
                props = module.cuda.get_device_properties(0)
                free, total = module.cuda.mem_get_info(0)
                result["device"].update(name=props.name, vram_total_gb=round(total / 1024**3, 2), vram_used_gb=round((total-free) / 1024**3, 2))
    except Exception as exc:
        result["dependencies"][name] = False
        result["dependency_errors"][name] = str(exc)[:1200]
        if name == "torch":
            result["device"]["cuda"] = False
print("TUNA_PROBE:" + json.dumps(result))
'''


class Studio:
    def __init__(self, base_dir=BASE_DIR, data_dir=None, probe=True):
        self.base_dir = Path(base_dir).resolve()
        self.data_dir = Path(data_dir or self.base_dir / "data").resolve()
        self.datasets_dir = self.data_dir / "datasets"
        self.runs_dir = self.data_dir / "runs"
        self.datasets_dir.mkdir(parents=True, exist_ok=True)
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.processes = {}
        self.export_processes = {}
        self.runs = {}
        self.closed = False
        self.chat = ChatSession(self.base_dir, self.data_dir)
        for state_file in self.runs_dir.glob("*/state.json"):
            try:
                run = json.loads(state_file.read_text(encoding="utf-8"))
                if not IDENTIFIER.fullmatch(run["id"]):
                    continue
                if run.get("status") in ("queued", "running", "stopping"):
                    run.update(status="failed", ended_at=now(), error="The server was interrupted. This run was not resumed; start a new run.")
                    atomic_json(state_file, run)
                try:
                    clean_import_work(state_file.parent)
                except OSError:
                    pass  # Windows may still be releasing an interrupted worker.
                if run.get("gguf", {}).get("status") in ("running", "stopping"):
                    run["gguf"].update(status="failed", error="GGUF export was interrupted. Start the export again.")
                    atomic_json(state_file, run)
                self.runs[run["id"]] = run
            except (OSError, ValueError, KeyError, TypeError):
                continue
        self.status = {"version": VERSION,
                       "device": {"name": "Checking local hardware…" if probe else "CPU", "vram_total_gb": 0, "vram_used_gb": 0, "cuda": False},
                       "dependencies": {name: False for name in DEPENDENCIES},
                       "dependency_errors": {}, "training_ready": False,
                       "missing": list(DEPENDENCIES[:-1]), "checking": probe,
                       "methods_ready": {"lora": False, "qlora": False, "tuna": False}, "setup_actions": [],
                       "offline": True, "python": sys.version.split()[0], "python_executable": sys.executable,
                       "data_directory": str(self.data_dir)}
        self.last_probe = 0
        if probe:
            threading.Thread(target=self.probe, daemon=True, name="readiness-probe").start()

    def probe(self):
        try:
            completed = subprocess.run([sys.executable, "-c", PROBE_CODE], capture_output=True,
                                       text=True, encoding="utf-8", errors="replace", timeout=180,
                                       env={**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
                                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            lines = [line for line in completed.stdout.splitlines() if line.startswith("TUNA_PROBE:")]
            if not lines:
                raise RuntimeError((completed.stderr.strip() or "Dependency check did not return a result.")[-1500:])
            result = json.loads(lines[-1][len("TUNA_PROBE:"):])
            with self.lock:
                self.status.update(result)
                missing = [name for name in DEPENDENCIES[:-1] if not result["dependencies"].get(name)]
                self.status.update(missing=missing, training_ready=not missing, checking=False)
                quantized_ready = not missing and result["device"]["cuda"] and result["dependencies"].get("bitsandbytes", False)
                self.status["methods_ready"] = {"lora": not missing, "qlora": quantized_ready, "tuna": quantized_ready}
                self.status["setup_actions"] = (["Install CUDA-enabled PyTorch for QLoRA and Tuna. The current PyTorch runtime cannot use CUDA."] if not result["device"]["cuda"] else [])
                if not result["dependencies"].get("bitsandbytes"):
                    self.status["setup_actions"].append("Install a compatible bitsandbytes build for QLoRA and Tuna.")
                self.last_probe = time.monotonic()
        except Exception as exc:
            with self.lock:
                self.status.update(checking=False, training_ready=False, probe_error=str(exc)[:1500])
                self.last_probe = time.monotonic()

    def get_status(self):
        with self.lock:
            result = copy.deepcopy(self.status)
            result["offline"] = self.settings()["offline"]
            result["active_run_id"] = next((r["id"] for r in self.runs.values() if r["status"] in ("queued", "running", "stopping")), None)
            if self.last_probe and time.monotonic() - self.last_probe > 600 and not self.status["checking"]:
                self.status["checking"] = True
                threading.Thread(target=self.probe, daemon=True).start()
            return result

    def dataset(self, identifier):
        if not IDENTIFIER.fullmatch(identifier):
            raise APIError("Dataset not found.", 404)
        try:
            return json.loads((self.datasets_dir / f"{identifier}.meta.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            raise APIError("Dataset not found.", 404)

    def list_datasets(self):
        with self.lock:
            items = []
            for path in self.datasets_dir.glob("*.meta.json"):
                try:
                    items.append(json.loads(path.read_text(encoding="utf-8")))
                except (OSError, ValueError):
                    continue
            return sorted(items, key=lambda item: item["created_at"], reverse=True)

    def add_dataset(self, payload):
        name = payload.get("name", "dataset.jsonl")
        if not isinstance(name, str) or not name.strip() or len(name) > 160 or any(c in name for c in "\r\n\x00/\\"):
            raise APIError("Choose a file name of 1–160 characters without path separators.")
        records = normalize_jsonl(payload.get("content"))
        content = "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n"
        identifier = uuid.uuid4().hex
        dataset = {"id": identifier, "name": name.strip(), "rows": len(records),
                   "size_bytes": len(content.encode("utf-8")), "created_at": now(), "preview": records[:3]}
        with self.lock:
            (self.datasets_dir / f"{identifier}.jsonl").write_text(content, encoding="utf-8")
            atomic_json(self.datasets_dir / f"{identifier}.meta.json", dataset)
        return dataset

    def delete_dataset(self, identifier):
        with self.lock:
            self.dataset(identifier)
            if any(run["config"].get("dataset_id") == identifier and run["status"] in ("queued", "running", "stopping") for run in self.runs.values()):
                raise APIError("This dataset is being used by an active run. Stop that run first.", 409)
            (self.datasets_dir / f"{identifier}.jsonl").unlink(missing_ok=True)
            (self.datasets_dir / f"{identifier}.meta.json").unlink(missing_ok=True)

    def validate_config(self, payload):
        method = payload.get("method", "qlora")
        if method not in ("qlora", "lora", "tuna"):
            raise APIError("Choose qlora, lora, or tuna as the training method.")
        name = payload.get("name", "Untitled run")
        if not isinstance(name, str) or not name.strip() or len(name) > 160:
            raise APIError("Run name must be between 1 and 160 characters.")
        model = payload.get("model", payload.get("model_path", ""))
        if not isinstance(model, str) or not model.strip() or len(model) > 1000 or "\x00" in model:
            raise APIError("Enter a local GGUF file, model directory, or Hugging Face model ID.")
        offline = payload.get("offline", self.settings()["offline"])
        if not isinstance(offline, bool):
            raise APIError("offline must be true or false.")
        dataset_id = payload.get("dataset_id", "")
        if not isinstance(dataset_id, str):
            raise APIError("Select a valid dataset.")
        self.dataset(dataset_id)
        prompt = payload.get("system_prompt", "")
        if not isinstance(prompt, str) or len(prompt) > 50000:
            raise APIError("The system prompt must be text, at most 50,000 characters.")
        if method == "tuna" and not prompt.strip():
            raise APIError("Tuna needs a system prompt to reinforce during training.")
        checkpointing = payload.get("gradient_checkpointing", True)
        if not isinstance(checkpointing, bool):
            raise APIError("gradient_checkpointing must be true or false.")
        model = model.strip()
        try:
            gguf_source = local_gguf_path(model)
        except ValueError as exc:
            raise APIError(str(exc)) from exc
        if Path(model).expanduser().is_dir():
            model_path = str(Path(model).expanduser().resolve())
        elif gguf_source:
            model_path = str(gguf_source)
        else:
            model_path = model  # A cached Hugging Face model ID is supported offline.
        config = {"name": name.strip(), "model": model, "model_path": model_path,
                  "method": method, "dataset_id": dataset_id,
                  "dataset_path": str(self.datasets_dir / f"{dataset_id}.jsonl"),
                  "system_prompt": prompt, "offline": offline,
                  "gradient_checkpointing": checkpointing}
        for key, default, minimum, maximum, integer in (
                ("epochs", 3, 0.01, 1000, False),
                ("learning_rate", 0.0002, 0.00000001, 0.1, False),
                ("batch_size", 1, 1, 128, True),
                ("gradient_accumulation", 8, 1, 1024, True),
                ("max_seq_length", 1024, 32, 131072, True),
                ("lora_rank", 16, 1, 512, True),
                ("lora_alpha", 32, 1, 2048, True),
                ("lora_dropout", 0, 0, 0.5, False),
                ("seed", 42, 0, 2147483647, True),
                ("max_steps", 0, 0, 10000000, True)):
            config[key] = finite_number(payload, key, default, minimum, maximum, integer)
        config["gradient_accumulation_steps"] = config["gradient_accumulation"]
        return config

    def _save_run(self, run):
        atomic_json(self.runs_dir / run["id"] / "state.json", run)

    def get_run(self, identifier):
        with self.lock:
            if identifier not in self.runs:
                raise APIError("Run not found.", 404)
            return copy.deepcopy(self.runs[identifier])

    def list_runs(self):
        with self.lock:
            return sorted((copy.deepcopy(run) for run in self.runs.values()), key=lambda item: item["created_at"], reverse=True)

    def load_chat(self, payload):
        identifier = payload.get("run_id")
        source = payload.get("source", "adapter")
        if not isinstance(identifier, str) or not IDENTIFIER.fullmatch(identifier):
            raise APIError("Choose a completed fine-tune.")
        if source not in ("adapter", "gguf"):
            raise APIError("Choose the trained adapter or its exported GGUF.")
        with self.lock:
            run = self.get_run(identifier)
            if run["status"] != "completed":
                raise APIError("Chat is available after training completes.", 409)
            if self.export_processes or self.processes:
                raise APIError("Wait for training or GGUF export to finish before loading chat.", 409)
            directory = self.runs_dir / identifier
            if source == "gguf":
                self._gguf_file(identifier)
            elif not (directory / "adapter" / "adapter_model.safetensors").is_file():
                raise APIError("The saved adapter is missing. Choose this run's GGUF export instead.", 404)
            return self.chat.load(directory, identifier, source, run.get("name", "Fine-tuned model"))

    def start_run(self, payload):
        with self.lock:
            config = self.validate_config(payload)
            if self.export_processes:
                raise APIError("A GGUF export is active. Wait for it to finish or stop the export before training.", 409)
            if any(run["status"] in ("queued", "running", "stopping") for run in self.runs.values()):
                raise APIError("A training run is already active. Stop it before starting another.", 409)
            if not self.status["training_ready"]:
                detail = "The local dependency check is still running." if self.status["checking"] else "Missing or incompatible dependencies: " + ", ".join(self.status["missing"])
                raise APIError(detail + " Open Setup for installation instructions.", 409)
            if config["method"] in ("qlora", "tuna"):
                if not self.status["device"]["cuda"] or not self.status["dependencies"].get("bitsandbytes"):
                    raise APIError("QLoRA and Tuna require a supported CUDA GPU and a working bitsandbytes installation. LoRA supports CPU training.", 409)
            trainer = self.base_dir / "trainer.py"
            if not trainer.is_file():
                raise APIError("trainer.py is missing from this installation.", 503)
            self.chat.release_for_job()
            identifier = uuid.uuid4().hex
            run_dir = self.runs_dir / identifier
            run_dir.mkdir()
            config.update(run_id=identifier, output_dir=str(run_dir))
            atomic_json(run_dir / "config.json", config)
            run = {"id": identifier, "name": config["name"], "model": config["model"],
                   "method": config["method"], "dataset_id": config["dataset_id"],
                   "status": "queued", "created_at": now(), "started_at": None,
                   "ended_at": None, "config": config, "metrics": [], "logs": [],
                   "error": None, "artifact_available": False}
            self.runs[identifier] = run
            self._save_run(run)
            try:
                process = subprocess.Popen([sys.executable, "-u", str(trainer), "--config", str(run_dir / "config.json"), "--output", str(run_dir)],
                                           cwd=str(self.base_dir), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                           text=True, encoding="utf-8", errors="replace", bufsize=1,
                                           env={**os.environ, "PYTHONUNBUFFERED": "1", "HF_HUB_OFFLINE": "1" if config["offline"] else "0", "TRANSFORMERS_OFFLINE": "1" if config["offline"] else "0", "HF_HUB_DISABLE_TELEMETRY": "1", "WANDB_DISABLED": "true"},
                                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            except OSError as exc:
                run.update(status="failed", error=str(exc), ended_at=now())
                self._save_run(run)
                raise APIError("Could not start the trainer: " + str(exc), 500)
            self.processes[identifier] = process
            run.update(status="running", started_at=now())
            self._save_run(run)
            threading.Thread(target=self._watch_run, args=(identifier, process), daemon=True, name=f"trainer-{identifier[:8]}").start()
            return copy.deepcopy(run)

    def _watch_run(self, identifier, process):
        completion_seen = False
        try:
            for line in process.stdout:
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                    if not isinstance(event, dict):
                        event = {"type": "log", "message": line}
                except (ValueError, RecursionError):
                    event = {"type": "log", "message": line}
                with self.lock:
                    run = self.runs[identifier]
                    kind = event.get("type", "log")
                    if kind in ("metric", "metrics"):
                        safe_event = {key: value for key, value in event.items() if isinstance(value, (str, int, float, bool, type(None))) and (not isinstance(value, float) or math.isfinite(value))}
                        safe_event["timestamp"] = now()
                        run["metrics"].append(safe_event)
                        run["metrics"] = run["metrics"][-10000:]
                    else:
                        message = str(event.get("message", line))[:12000]
                        run["logs"].append({"timestamp": now(), "level": "error" if kind == "error" else "info", "message": message})
                        run["logs"] = run["logs"][-1000:]
                    if kind == "complete":
                        completion_seen = True
                        run["result"] = event
                        run["summary"] = {key: value for key, value in event.items() if key != "type"}
                    if kind == "error":
                        run["error"] = str(event.get("message", "Training failed."))[:4000]
                    self._save_run(run)
            code = process.wait()
            with self.lock:
                run = self.runs[identifier]
                if run["status"] == "stopping":
                    run.update(status="stopped", error=None)
                elif code == 0 and completion_seen and not run["error"]:
                    run.update(status="completed", artifact_available=bool(self._artifact_files(identifier)))
                else:
                    run.update(status="failed", error=run["error"] or (f"Trainer exited with code {code}. See the run log." if code else "Trainer exited without reporting a completed adapter."))
                run.update(ended_at=now(), exit_code=code)
                self._save_run(run)
                self.processes.pop(identifier, None)
        except Exception as exc:
            with self.lock:
                run = self.runs[identifier]
                run.update(status="failed", error=f"Could not monitor training: {exc}", ended_at=now())
                self._save_run(run)
                self.processes.pop(identifier, None)
            if process.poll() is None:
                process.kill()
        finally:
            if process.stdout:
                process.stdout.close()
            try:
                clean_import_work(self.runs_dir / identifier)
            except OSError:
                pass

    def stop_run(self, identifier):
        with self.lock:
            run = self.runs.get(identifier)
            if not run:
                raise APIError("Run not found.", 404)
            process = self.processes.get(identifier)
            if not process or run["status"] not in ("running", "queued", "stopping"):
                raise APIError("This run is no longer active.", 409)
            run["status"] = "stopping"
            self._save_run(run)
            try:
                process.terminate()
            except OSError:
                pass
            def reap():
                try:
                    process.wait(timeout=8)
                except subprocess.TimeoutExpired:
                    process.kill()
            threading.Thread(target=reap, daemon=True).start()
            return copy.deepcopy(run)

    def _artifact_files(self, identifier):
        root = (self.runs_dir / identifier).resolve()
        excluded = {"state.json", "config.json"}
        files = []
        for path in root.rglob("*"):
            if any(part.startswith("gguf-import-work-") for part in path.relative_to(root).parts):
                continue
            if path.relative_to(root).parts[0] == "gguf":
                continue
            if path.is_file() and path.resolve().is_relative_to(root) and path.name not in excluded and not path.name.endswith(".tmp"):
                files.append(path)
        return files

    def _gguf_file(self, identifier):
        run = self.get_run(identifier)
        export = run.get("gguf", {})
        if export.get("status") != "completed":
            raise APIError("GGUF download is available after export completes.", 409)
        root = (self.runs_dir / identifier / "gguf").resolve()
        filename = export.get("result", {}).get("filename", "")
        path = (root / filename).resolve()
        if path.parent != root or path.suffix != ".gguf" or not path.is_file():
            raise APIError("The exported GGUF file is missing.", 404)
        return path

    def start_gguf(self, identifier, payload):
        output_type = payload.get("format", "q8_0")
        if output_type not in ("q8_0", "f16"):
            raise APIError("Choose q8_0 or f16 for GGUF export.")
        with self.lock:
            run = self.runs.get(identifier)
            if not run:
                raise APIError("Run not found.", 404)
            if run["status"] != "completed":
                raise APIError("Complete training before exporting GGUF.", 409)
            current = run.get("gguf", {})
            if current.get("status") == "completed" and current.get("format") == output_type:
                try:
                    self._gguf_file(identifier)
                    return copy.deepcopy(run)
                except APIError:
                    pass
            if identifier in self.export_processes:
                if current.get("format") == output_type:
                    return copy.deepcopy(run)
                raise APIError("This run is already being exported.", 409)
            if self.export_processes or any(r["status"] in ("queued", "running", "stopping") for r in self.runs.values()):
                raise APIError("Wait for the active training run or GGUF export to finish first.", 409)
            worker = self.base_dir / "gguf_export.py"
            self.chat.release_for_job()
            if not worker.is_file():
                raise APIError("The GGUF exporter is missing from this installation.", 503)
            run["gguf"] = {"status": "running", "format": output_type, "stage": "preparing", "started_at": now(),
                           "message": "Preparing the cached model and trained adapter.", "logs": [], "error": None}
            self._save_run(run)
            try:
                process = subprocess.Popen([sys.executable, "-u", str(worker), "--run-dir", str(self.runs_dir / identifier), "--format", output_type],
                    cwd=str(self.base_dir), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, encoding="utf-8", errors="replace", bufsize=1,
                    env={**os.environ, "PYTHONUNBUFFERED": "1", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1"},
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            except OSError as exc:
                run["gguf"].update(status="failed", error=str(exc))
                self._save_run(run)
                raise APIError("Could not start GGUF export: " + str(exc), 500)
            self.export_processes[identifier] = process
            threading.Thread(target=self._watch_gguf, args=(identifier, process), daemon=True, name=f"gguf-{identifier[:8]}").start()
            return copy.deepcopy(run)

    def _watch_gguf(self, identifier, process):
        complete = False
        try:
            for line in process.stdout:
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                    if not isinstance(event, dict):
                        event = {}
                except (ValueError, RecursionError):
                    event = {}
                with self.lock:
                    run = self.runs[identifier]
                    export = run["gguf"]
                    kind = event.get("type")
                    if kind == "export_progress":
                        for key in ("stage", "message", "completed", "total"):
                            if key in event:
                                export[key] = event[key]
                    elif kind == "export_complete":
                        complete = True
                        export["result"] = {key: value for key, value in event.items() if key != "type"}
                    elif kind == "export_error":
                        export["error"] = str(event.get("message", "GGUF export failed"))[:4000]
                    else:
                        export["logs"].append(line[:3000])
                        export["logs"] = export["logs"][-60:]
                    export["updated_at"] = now()
                    self._save_run(run)
            code = process.wait()
            with self.lock:
                run = self.runs[identifier]
                export = run["gguf"]
                if export["status"] == "stopping":
                    export.update(status="stopped", error=None, message="GGUF export stopped.")
                elif code == 0 and complete and not export.get("error"):
                    export.update(status="completed", stage="completed", message="Your standalone GGUF is ready.")
                    try:
                        path = self._gguf_file(identifier)
                        with path.open("rb") as handle:
                            if handle.read(4) != b"GGUF":
                                raise ValueError("Invalid GGUF header")
                    except (APIError, OSError, ValueError) as exc:
                        export.update(status="failed", error=str(exc))
                else:
                    export.update(status="failed", error=export.get("error") or f"GGUF worker exited with code {code} without a completed export.")
                export["ended_at"] = now()
                self._cleanup_gguf_scratch(identifier)
                self._save_run(run)
                self.export_processes.pop(identifier, None)
        except Exception as exc:
            with self.lock:
                self.runs[identifier]["gguf"].update(status="failed", error=str(exc), ended_at=now())
                self._cleanup_gguf_scratch(identifier)
                self._save_run(self.runs[identifier])
                self.export_processes.pop(identifier, None)
            if process.poll() is None:
                process.kill()
        finally:
            if process.stdout:
                process.stdout.close()

    def _cleanup_gguf_scratch(self, identifier):
        root = (self.runs_dir / identifier / "gguf").resolve()
        run_root = (self.runs_dir / identifier).resolve()
        if not root.is_relative_to(run_root) or not root.is_dir():
            return
        for path in root.glob("merge-work-*"):
            resolved = path.resolve()
            if resolved.is_relative_to(root) and path.is_dir() and path.name.startswith("merge-work-"):
                try:
                    shutil.rmtree(resolved)
                except OSError:
                    pass  # A later export can retry cleanup after Windows releases the mapping.
        for path in root.glob("*.gguf.partial"):
            if path.is_file() and path.resolve().parent == root:
                try:
                    path.unlink()
                except OSError:
                    pass

    def stop_gguf(self, identifier):
        with self.lock:
            process = self.export_processes.get(identifier)
            if process is None:
                raise APIError("No GGUF export is active for this run.", 409)
            run = self.runs[identifier]
            run["gguf"]["status"] = "stopping"
            self._save_run(run)
            process.terminate()
            return copy.deepcopy(run)

    def gguf_companion(self, identifier):
        model = self._gguf_file(identifier)
        handle = tempfile.TemporaryFile()
        with zipfile.ZipFile(handle, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for filename in ("Modelfile", "system-prompt.txt", "policy.json", "README.txt", "export_manifest.json"):
                path = model.parent / filename
                if path.is_file() and path.resolve().parent == model.parent:
                    archive.write(path, filename)
        size = handle.tell()
        handle.seek(0)
        return handle, size

    def export_run(self, identifier):
        run = self.get_run(identifier)
        if run["status"] != "completed":
            raise APIError("Adapters can be exported after a training run completes.", 409)
        files = self._artifact_files(identifier)
        if not files:
            raise APIError("This run has no adapter files to export.", 404)
        handle = tempfile.TemporaryFile()
        with zipfile.ZipFile(handle, "w", compression=zipfile.ZIP_STORED) as archive:
            for path in files:
                archive.write(path, str(path.relative_to(self.runs_dir / identifier)))
        size = handle.tell()
        handle.seek(0)
        return handle, size

    def settings(self, payload=None):
        path = self.data_dir / "settings.json"
        with self.lock:
            try:
                settings = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                settings = {"offline": False, "default_model": "", "theme": "light"}
            if payload is not None:
                offline = payload.get("offline", settings.get("offline", False))
                if not isinstance(offline, bool):
                    raise APIError("offline must be true or false.")
                model = payload.get("default_model", settings.get("default_model", ""))
                theme = payload.get("theme", settings.get("theme", "light"))
                if not isinstance(model, str) or len(model) > 1000 or theme not in ("light", "dark", "system"):
                    raise APIError("Invalid settings.")
                settings = {"offline": offline, "default_model": model, "theme": theme}
                atomic_json(path, settings)
            return settings

    def close(self):
        self.closed = True
        self.chat.close()
        with self.lock:
            active = list(self.processes.items())
            exports = list(self.export_processes.items())
        for identifier, process in exports:
            try:
                self.stop_gguf(identifier)
                process.wait(timeout=10)
            except (APIError, OSError, subprocess.TimeoutExpired):
                if process.poll() is None:
                    process.kill()
        for identifier, process in active:
            try:
                self.stop_run(identifier)
                process.wait(timeout=10)
            except (APIError, OSError, subprocess.TimeoutExpired):
                if process.poll() is None:
                    process.kill()


class Handler(BaseHTTPRequestHandler):
    server_version = "TunaStudio/" + VERSION
    sys_version = ""
    protocol_version = "HTTP/1.1"

    def log_message(self, format, *args):
        if getattr(self.server, "verbose", False):
            super().log_message(format, *args)

    @property
    def studio(self):
        return self.server.studio

    def security_headers(self):
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; font-src 'self'; connect-src 'self'; frame-ancestors 'none'; base-uri 'self'; form-action 'self'")

    def json_response(self, data, status=200):
        encoded = json.dumps(data, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Cache-Control", "no-store")
        self.security_headers()
        if self.close_connection:
            self.send_header("Connection", "close")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(encoded)

    def check_origin(self):
        host = self.headers.get("Host", "")
        allowed = {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}
        if self.server.server_port == 80:
            allowed.update(("127.0.0.1", "localhost"))
        if host.lower() not in allowed:
            raise APIError("This app accepts requests only from its local address.", 403)
        origin = self.headers.get("Origin")
        if origin and origin != f"http://{host}":
            raise APIError("Cross-origin requests are not allowed.", 403)
        if self.headers.get("Sec-Fetch-Site") == "cross-site":
            raise APIError("Cross-site requests are not allowed.", 403)

    def read_json(self):
        if self.headers.get("Transfer-Encoding"):
            raise APIError("Use a request with Content-Length.", 400)
        if self.headers.get_content_type() != "application/json":
            raise APIError("Use application/json for this request.", 415)
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            raise APIError("Invalid Content-Length.")
        if length <= 0:
            raise APIError("A JSON request body is required.")
        if length > MAX_BODY:
            raise APIError("Upload exceeds the 25 MB request limit.", 413)
        self.connection.settimeout(30)
        body = self.rfile.read(length)
        if len(body) != length:
            raise APIError("The request body was incomplete.")
        try:
            value = json.loads(body.decode("utf-8"), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
        except (ValueError, UnicodeError, RecursionError):
            raise APIError("The request body must be valid UTF-8 JSON.")
        if not isinstance(value, dict):
            raise APIError("The request body must be a JSON object.")
        return value

    def do_GET(self):
        self.dispatch()

    def do_HEAD(self):
        self.dispatch()

    def do_POST(self):
        self.dispatch()

    def do_PUT(self):
        self.dispatch()

    def do_DELETE(self):
        self.dispatch()

    def do_OPTIONS(self):
        self.close_connection = True
        self.json_response({"error": "Cross-origin requests are not supported."}, 403)

    def dispatch(self):
        try:
            self.check_origin()
            raw_path = urlsplit(self.path).path
            path = unquote(raw_path)
            if "\x00" in path or "\\" in path or ".." in path.split("/"):
                raise APIError("Invalid path.", 400)
            if path.startswith("/api/"):
                self.api(path)
            elif self.command in ("GET", "HEAD"):
                self.static(path)
            else:
                raise APIError("Endpoint not found.", 404)
        except (APIError, ChatError) as exc:
            self.close_connection = True
            self.json_response({"error": str(exc)}, exc.status)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except (TimeoutError, OSError) as exc:
            self.close_connection = True
            self.json_response({"error": "Local file or connection error: " + str(exc)[:500]}, 500)
        except Exception:
            self.close_connection = True
            self.json_response({"error": "An unexpected local server error occurred."}, 500)

    def api(self, path):
        method = "GET" if self.command == "HEAD" else self.command
        segments = path.strip("/").split("/")
        if path == "/api/status" and method == "GET":
            return self.json_response(self.studio.get_status())
        if path == "/api/chat" and method == "GET":
            return self.json_response({"chat": self.studio.chat.snapshot()})
        if len(segments) == 4 and segments[:3] == ["api", "chat", "replies"] and method == "GET":
            if not IDENTIFIER.fullmatch(segments[3]):
                raise APIError("Invalid chat request identifier.")
            return self.json_response({"reply": self.studio.chat.reply(segments[3])})
        if path == "/api/chat/load" and method == "POST":
            return self.json_response({"chat": self.studio.load_chat(self.read_json())}, 202)
        if path == "/api/chat/message" and method == "POST":
            return self.json_response({"chat": self.studio.chat.generate(self.read_json())}, 202)
        if path == "/api/chat/stop" and method == "POST":
            self.read_json()
            return self.json_response({"chat": self.studio.chat.stop()})
        if path == "/api/chat/unload" and method == "POST":
            self.read_json()
            return self.json_response({"chat": self.studio.chat.unload()})
        if path == "/api/datasets":
            if method == "GET":
                return self.json_response({"datasets": self.studio.list_datasets()})
            if method == "POST":
                return self.json_response({"dataset": self.studio.add_dataset(self.read_json())}, 201)
        if len(segments) == 3 and segments[:2] == ["api", "datasets"]:
            if method == "GET":
                return self.json_response({"dataset": self.studio.dataset(segments[2])})
            if method == "DELETE":
                self.studio.delete_dataset(segments[2])
                return self.json_response({"deleted": True})
        if path == "/api/runs":
            if method == "GET":
                return self.json_response({"runs": self.studio.list_runs()})
            if method == "POST":
                return self.json_response({"run": self.studio.start_run(self.read_json())}, 201)
        if len(segments) >= 3 and segments[:2] == ["api", "runs"]:
            identifier = segments[2]
            if not IDENTIFIER.fullmatch(identifier):
                raise APIError("Run not found.", 404)
            if len(segments) == 3 and method == "GET":
                return self.json_response({"run": self.studio.get_run(identifier)})
            if len(segments) == 4 and segments[3] == "stop" and method == "POST":
                self.read_json()
                return self.json_response({"run": self.studio.stop_run(identifier)})
            if len(segments) == 4 and segments[3] == "gguf":
                if method == "POST":
                    return self.json_response({"run": self.studio.start_gguf(identifier, self.read_json())}, 201)
                if method == "GET":
                    file = self.studio._gguf_file(identifier)
                    self.send_response(200)
                    self.send_header("Content-Type", "application/octet-stream")
                    self.send_header("Content-Disposition", f'attachment; filename="{file.name}"')
                    self.send_header("Content-Length", str(file.stat().st_size))
                    self.security_headers()
                    self.end_headers()
                    if self.command != "HEAD":
                        with file.open("rb") as handle:
                            while chunk := handle.read(1024 * 1024):
                                self.wfile.write(chunk)
                    return
            if len(segments) == 5 and segments[3:] == ["gguf", "stop"] and method == "POST":
                self.read_json()
                return self.json_response({"run": self.studio.stop_gguf(identifier)})
            if len(segments) == 5 and segments[3:] == ["gguf", "companion"] and method == "GET":
                archive, size = self.studio.gguf_companion(identifier)
                try:
                    self.send_response(200)
                    self.send_header("Content-Type", "application/zip")
                    self.send_header("Content-Disposition", 'attachment; filename="tuna-gguf-ollama-files.zip"')
                    self.send_header("Content-Length", str(size))
                    self.security_headers()
                    self.end_headers()
                    if self.command != "HEAD":
                        while chunk := archive.read(1024 * 1024):
                            self.wfile.write(chunk)
                finally:
                    archive.close()
                return
            if len(segments) == 4 and segments[3] in ("export", "artifacts") and method == "GET":
                archive, size = self.studio.export_run(identifier)
                try:
                    self.send_response(200)
                    self.send_header("Content-Type", "application/zip")
                    self.send_header("Content-Disposition", f'attachment; filename="tuna-{identifier[:8]}-adapter.zip"')
                    self.send_header("Content-Length", str(size))
                    self.security_headers()
                    self.end_headers()
                    if self.command != "HEAD":
                        while chunk := archive.read(1024 * 1024):
                            self.wfile.write(chunk)
                finally:
                    archive.close()
                return
        if path == "/api/settings":
            if method == "GET":
                return self.json_response({"settings": self.studio.settings()})
            if method == "PUT":
                return self.json_response({"settings": self.studio.settings(self.read_json())})
        raise APIError("Endpoint not found.", 404)

    def static(self, path):
        root = (self.studio.base_dir / "web").resolve()
        requested = (root / (path.lstrip("/") or "index.html")).resolve()
        if not requested.is_relative_to(root) or not requested.is_file():
            raise APIError("File not found.", 404)
        content_type = mimetypes.guess_type(str(requested))[0] or "application/octet-stream"
        if requested.suffix in (".js", ".mjs"):
            content_type = "text/javascript"
        if content_type.startswith("text/") or content_type == "application/json":
            content_type += "; charset=utf-8"
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(requested.stat().st_size))
        self.send_header("Cache-Control", "no-cache")
        self.security_headers()
        self.end_headers()
        if self.command != "HEAD":
            with requested.open("rb") as handle:
                while chunk := handle.read(1024 * 1024):
                    self.wfile.write(chunk)


def make_server(studio, port=8765, host="127.0.0.1", verbose=False):
    if host not in ("127.0.0.1", "localhost"):
        raise ValueError("Tuna Studio must bind to 127.0.0.1 or localhost.")
    server = ThreadingHTTPServer((host, port), Handler)
    server.studio = studio
    server.verbose = verbose
    server.daemon_threads = True
    return server


def main():
    parser = argparse.ArgumentParser(description="Run Tuna Studio locally. No model or data leaves your machine.")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--host", choices=("127.0.0.1", "localhost"), default="127.0.0.1")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    studio = Studio(data_dir=args.data_dir)
    try:
        server = make_server(studio, args.port, args.host, args.verbose)
    except OSError as exc:
        print(f"Could not start Tuna Studio: {exc}", file=sys.stderr)
        return 1
    print(f"Tuna Studio is running at http://{args.host}:{server.server_port}", flush=True)
    print(f"Local data: {studio.data_dir}", flush=True)
    print("Press Ctrl+C to stop the server and any active training run.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        studio.close()
        server.server_close()
    return 0


if __name__ == "__main__":
    from app_runtime import use_app_environment
    use_app_environment(__file__)
    raise SystemExit(main())
