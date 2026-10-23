"""Dependency-free supervision of one local inference worker."""
from __future__ import annotations

import copy
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import uuid


class ChatError(Exception):
    def __init__(self, message, status=400):
        super().__init__(message)
        self.status = status


def validate_message_request(payload):
    messages = payload.get("messages")
    if not isinstance(messages, list) or not 1 <= len(messages) <= 199:
        raise ChatError("Send between 1 and 199 conversation messages.")
    clean, size = [], 0
    for index, message in enumerate(messages):
        if not isinstance(message, dict) or message.get("role") != ("user" if index % 2 == 0 else "assistant"):
            raise ChatError("Chat messages must alternate user and assistant. Caller system/developer messages are not allowed.")
        content = message.get("content")
        if not isinstance(content, str) or not content.strip() or len(content) > 20000:
            raise ChatError("Each message must contain 1–20,000 characters.")
        size += len(content)
        clean.append({"role": message["role"], "content": content})
    if clean[-1]["role"] != "user":
        raise ChatError("The conversation must end with a user message.")
    if size > 100000:
        raise ChatError("This conversation is too long. Start a new chat.")
    tokens = payload.get("max_new_tokens", 256)
    if type(tokens) is not int or not 1 <= tokens <= 2048:
        raise ChatError("Response length must be between 1 and 2,048 tokens.")
    temperature = payload.get("temperature", 0.7)
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or not math.isfinite(temperature) or not 0 <= temperature <= 2:
        raise ChatError("Temperature must be between 0 and 2.")
    return {"messages": clean, "max_new_tokens": tokens, "temperature": float(temperature)}


class ChatSession:
    def __init__(self, base_dir, data_dir, idle_seconds=300):
        self.base_dir, self.data_dir = Path(base_dir), Path(data_dir)
        self.lock = threading.RLock()
        self.process = None
        self.closed = threading.Event()
        self.idle_seconds = idle_seconds
        self.last_used = time.monotonic()
        self.instance_id, self.revision = uuid.uuid4().hex, 0
        self.replies = {}
        self.state = {"status": "unloaded", "run_id": None, "source": None,
                      "request_id": None, "text": "", "error": None,
                      "message": "Choose a fine-tune to start chatting.", "logs": []}
        self.idle_started = False

    def snapshot(self):
        with self.lock:
            self.revision += 1
            return {**copy.deepcopy(self.state), "instance_id": self.instance_id, "revision": self.revision}

    def load(self, run_dir, run_id, source, name):
        with self.lock:
            if self.closed.is_set():
                raise ChatError("The local server is shutting down.", 503)
            if self.state["status"] in ("loading", "generating"):
                raise ChatError("Finish or stop the current chat operation first.", 409)
            if self.state["status"] == "ready" and self.state["run_id"] == run_id and self.state["source"] == source:
                return self.snapshot()
            worker = self.base_dir / "chat_worker.py"
            if not worker.is_file():
                raise ChatError("The local chat worker is missing from this installation.", 503)
            self.unload()
            self.state = {"status": "loading", "run_id": run_id, "source": source,
                          "name": name, "request_id": None, "text": "", "error": None,
                          "message": "Loading your fine-tune locally…", "logs": []}
            try:
                process = subprocess.Popen([sys.executable, "-u", str(worker), "--run-dir", str(run_dir),
                    "--source", source, "--cache-dir", str(self.data_dir / "models"),
                    "--work-dir", str(self.data_dir / "chat-work")], cwd=self.base_dir,
                    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, encoding="utf-8", errors="replace", bufsize=1,
                    env={**os.environ,"PYTHONUNBUFFERED":"1","PYTHONIOENCODING":"utf-8","HF_HUB_OFFLINE":"1",
                         "TRANSFORMERS_OFFLINE":"1","HF_HUB_DISABLE_TELEMETRY":"1","TOKENIZERS_PARALLELISM":"false"},
                    creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0))
            except OSError as exc:
                self.state.update(status="error",error=str(exc),message="Could not start local chat.")
                raise ChatError("Could not start local chat: " + str(exc), 500) from exc
            self.process = process
            self.last_used = time.monotonic()
            threading.Thread(target=self._watch,args=(process,),daemon=True,name="chat-worker").start()
            if not self.idle_started:
                self.idle_started = True
                threading.Thread(target=self._idle_loop,daemon=True,name="chat-idle").start()
            return self.snapshot()

    def _send(self, value):
        try:
            self.process.stdin.write(json.dumps(value, ensure_ascii=False) + "\n")
            self.process.stdin.flush()
        except (OSError, AttributeError, ValueError) as exc:
            self.state.update(status="error",error="The chat worker disconnected. Load the model again.")
            raise ChatError(self.state["error"], 503) from exc

    def generate(self, payload):
        request = validate_message_request(payload)
        with self.lock:
            if not self.process or self.process.poll() is not None or self.state["status"] != "ready":
                raise ChatError("Load a model and wait until it is ready before sending a message.", 409)
            if payload.get("run_id") is not None and (payload["run_id"] != self.state["run_id"] or payload.get("source", "adapter") != self.state["source"]):
                raise ChatError("The loaded model changed in another chat. Load your selected fine-tune again; your draft is kept.", 409)
            identifier = uuid.uuid4().hex
            self.state.update(status="generating",request_id=identifier,text="",error=None,
                              usage=None,finish_reason=None,stop_requested=False,message="Your model is replying…")
            self.last_used = time.monotonic()
            self._send({"type":"generate","request_id":identifier,**request})
            return self.snapshot()

    def stop(self):
        with self.lock:
            if self.state["status"] != "generating":
                raise ChatError("There is no active reply to stop.", 409)
            self._send({"type":"stop"})
            self.state.update(stop_requested=True,message="Stopping the reply…")
            return self.snapshot()

    def _remember_reply(self):
        identifier = self.state.get("request_id")
        if identifier:
            self.replies[identifier] = {key:copy.deepcopy(self.state.get(key)) for key in
                ("request_id","run_id","source","text","error","usage","finish_reason")}
            while len(self.replies) > 16:
                self.replies.pop(next(iter(self.replies)))

    def reply(self, identifier):
        with self.lock:
            return copy.deepcopy(self.replies.get(identifier))

    def unload(self, idle=False):
        with self.lock:
            if idle and (self.state["status"] != "ready" or time.monotonic() - self.last_used < self.idle_seconds):
                return self.snapshot()
            process, self.process = self.process, None
            if self.state["status"] == "generating":
                self.state.update(finish_reason="unloaded", error=None)
                self._remember_reply()
            if process:
                try:
                    if process.poll() is None:
                        if os.name == "nt":
                            # Windows' venv executable is a launcher. Reap its
                            # interpreter child too, including a cancelled load.
                            subprocess.run(["taskkill","/PID",str(process.pid),"/T","/F"],
                                stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=10,
                                creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0))
                        else:
                            process.terminate()
                    if process.stdin:
                        process.stdin.close()
                    if process.poll() is None:
                        process.terminate()
                    process.wait(timeout=5)
                except (OSError, ValueError, subprocess.TimeoutExpired):
                    if process.poll() is None:
                        process.kill()
                        process.wait(timeout=5)
            self.state.update(status="unloaded",error=None,finish_reason="unloaded",
                message="Model unloaded after five minutes of inactivity. Load it again to continue." if idle else "Model unloaded. Your chat is kept in this browser.")
            # The terminated worker no longer owns Windows mappings.
            from gguf_import import clean_import_work
            try:
                clean_import_work(self.data_dir / "chat-work")
            except OSError:
                pass
            return self.snapshot()

    def release_for_job(self):
        with self.lock:
            if self.state["status"] in ("loading", "generating"):
                raise ChatError("Finish or stop chat, then unload its model before starting training or export.", 409)
            if self.process:
                self.unload()

    def _watch(self, process):
        try:
            for line in process.stdout:
                try:
                    event = json.loads(line)
                    if not isinstance(event, dict):
                        continue
                except ValueError:
                    with self.lock:
                        if self.process is process:
                            self.state["logs"] = (self.state["logs"] + [line.strip()[:2000]])[-12:]
                    continue
                with self.lock:
                    if self.process is not process:
                        continue
                    kind = event.get("type")
                    if event.get("request_id") and event["request_id"] != self.state.get("request_id"):
                        continue
                    if kind == "progress":
                        self.state["message"] = str(event.get("message", "Loading…"))[:500]
                    elif kind == "ready":
                        self.state.update({key:event.get(key) for key in ("device","quantized","policy_bound","context_tokens",
                            "engine","precision","adapter_merged","optimizations","allocated_vram_mb")})
                        self.state.update(status="ready",message="Ready to chat.",error=None)
                        self.last_used = time.monotonic()
                    elif kind == "delta" and self.state["status"] == "generating":
                        self.state["text"] = (self.state["text"] + str(event.get("text","")))[:100000]
                    elif kind in ("reply_complete", "reply_error"):
                        if isinstance(event.get("text"),str):
                            self.state["text"] = event["text"][:100000]
                        self.state.update(status="ready",error=event.get("error"),usage=event.get("usage"),
                                          finish_reason=event.get("finish_reason"),stop_requested=False,
                                          message="Ready to chat." if kind == "reply_complete" else "Your message could not be completed.")
                        self._remember_reply()
                        self.last_used = time.monotonic()
                    elif kind == "fatal_error":
                        self.state.update(status="error",error=str(event.get("message","Model loading failed."))[:2000])
            code = process.wait()
            with self.lock:
                if self.process is process:
                    self.process = None
                    self.state.update(status="error",error=self.state.get("error") or f"The local chat worker exited ({code}). Load the model again.")
        finally:
            if process.stdout:
                process.stdout.close()
            if process.stdin:
                try:
                    process.stdin.close()
                except OSError:
                    pass

    def _idle_loop(self):
        while not self.closed.wait(5):
            self.unload(idle=True)

    def close(self):
        self.closed.set()
        self.unload()
