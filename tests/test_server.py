"""API, persistence and process lifecycle checks; no ML packages required."""
import http.client
import io
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
import zipfile
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from server import APIError, MAX_BODY, Studio, atomic_json, make_server, normalize_jsonl

EXAMPLE = {"messages": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello!"}]}


class DatasetTests(unittest.TestCase):
    def test_chat_and_prompt_completion_normalize(self):
        content = "\ufeff" + json.dumps(EXAMPLE) + "\n\n" + json.dumps({"prompt": "Q", "completion": "A"})
        records = normalize_jsonl(content)
        self.assertEqual(len(records), 2)
        self.assertEqual(records[1]["messages"][1], {"role": "assistant", "content": "A"})

    def test_errors_identify_line_and_invalid_role(self):
        with self.assertRaisesRegex(APIError, "Line 2"):
            normalize_jsonl(json.dumps(EXAMPLE) + "\n{broken")
        with self.assertRaisesRegex(APIError, "role"):
            normalize_jsonl(json.dumps({"messages": [{"role": "tool", "content": "X"}, {"role": "assistant", "content": "A"}]}))

    def test_incomplete_empty_and_nontext_examples_rejected(self):
        for value in ("", "\n", "[]", '{}', '{"prompt":"q","completion":null}', '{"messages":[{"role":"system","content":"s"},{"role":"assistant","content":"a"}]}'):
            with self.subTest(value=value), self.assertRaises(APIError):
                normalize_jsonl(value)


class APITests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / "web").mkdir()
        (self.root / "web" / "index.html").write_text("<h1>Tuna</h1>", encoding="utf-8")
        self.studio = Studio(self.root, probe=False)
        self.server = make_server(self.studio, port=0)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.port = self.server.server_port

    def tearDown(self):
        self.studio.close()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        self.temp.cleanup()

    def request(self, method, path, body=None, headers=None, raw=False):
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        request_headers = {"Content-Type": "application/json"}
        request_headers.update(headers or {})
        payload = json.dumps(body).encode("utf-8") if body is not None else None
        connection.request(method, path, payload, request_headers)
        response = connection.getresponse()
        data = response.read()
        result = (response.status, data if raw else json.loads(data), dict(response.getheaders()))
        connection.close()
        return result

    def upload(self):
        status, body, _ = self.request("POST", "/api/datasets", {"name": "support.jsonl", "content": json.dumps(EXAMPLE)})
        self.assertEqual(status, 201)
        return body["dataset"]

    def config(self, identifier, **changes):
        return {"model": "cached/model", "dataset_id": identifier, "method": "lora", "name": "Test", **changes}

    def ready(self):
        self.studio.status.update(training_ready=True, missing=[], dependencies={name: True for name in ("torch", "transformers", "peft", "accelerate", "bitsandbytes")})

    def wait_run(self, identifier, state):
        deadline = time.monotonic() + 6
        while time.monotonic() < deadline:
            run = self.studio.get_run(identifier)
            if run["status"] == state:
                return run
            time.sleep(0.03)
        self.fail(f"Expected {state}, got {self.studio.get_run(identifier)}")

    def test_dataset_crud_and_persistence(self):
        item = self.upload()
        self.assertEqual(item["rows"], 1)
        self.assertEqual(item["preview"], [EXAMPLE])
        self.assertEqual(self.request("GET", "/api/datasets")[1]["datasets"][0]["id"], item["id"])
        reloaded = Studio(self.root, probe=False)
        self.assertEqual(reloaded.dataset(item["id"])["name"], "support.jsonl")
        self.assertEqual(self.request("GET", "/api/datasets/" + item["id"])[0], 200)
        self.assertEqual(self.request("DELETE", "/api/datasets/" + item["id"])[0], 200)
        self.assertEqual(self.request("GET", "/api/datasets/" + item["id"])[0], 404)

    def test_dataset_name_cannot_be_a_path(self):
        status, body, _ = self.request("POST", "/api/datasets", {"name": "../outside.jsonl", "content": json.dumps(EXAMPLE)})
        self.assertEqual(status, 400)
        self.assertIn("file name", body["error"])

    def test_cross_origin_and_dns_rebinding_rejected(self):
        for headers in ({"Origin": "https://attacker.example"}, {"Host": f"attacker.example:{self.port}"}, {"Sec-Fetch-Site": "cross-site"}, {"Origin": "null"}):
            with self.subTest(headers=headers):
                self.assertEqual(self.request("GET", "/api/status", headers=headers)[0], 403)
        self.assertEqual(self.request("GET", "/api/status", headers={"Origin": f"http://127.0.0.1:{self.port}"})[0], 200)

    def test_traversal_and_private_files_not_served(self):
        (self.root / "secret.txt").write_text("secret", encoding="utf-8")
        for path in ("/../secret.txt", "/%2e%2e/secret.txt", "/%5c..%5csecret.txt", "/data/settings.json", "/server.py"):
            with self.subTest(path=path):
                self.assertIn(self.request("GET", path)[0], (400, 404))

    def test_no_directory_listing_and_local_static_headers(self):
        (self.root / "web" / "empty").mkdir()
        status, body, headers = self.request("GET", "/", raw=True)
        self.assertEqual(status, 200)
        self.assertIn(b"Tuna", body)
        self.assertEqual(headers["X-Content-Type-Options"], "nosniff")
        self.assertEqual(headers["X-Frame-Options"], "DENY")
        self.assertEqual(self.request("GET", "/empty/")[0], 404)

    def test_json_body_and_upload_limits(self):
        self.assertEqual(self.request("POST", "/api/datasets", [], headers={"Content-Type": "text/plain"})[0], 415)
        self.assertEqual(self.request("POST", "/api/datasets", ["not", "object"])[0], 400)
        self.assertEqual(self.request("POST", "/api/datasets", headers={"Content-Length": str(MAX_BODY + 1)})[0], 413)

    def test_missing_dependencies_never_create_a_run(self):
        dataset = self.upload()
        status, body, _ = self.request("POST", "/api/runs", self.config(dataset["id"]))
        self.assertEqual(status, 409)
        self.assertIn("dependencies", body["error"])
        self.assertEqual(self.studio.list_runs(), [])

    def test_cpu_runtime_rejects_quantization_even_with_gpu_detected(self):
        dataset = self.upload()
        self.ready()
        self.studio.status["device"].update(name="NVIDIA GPU", detected_gpu=True, cuda=False)
        for method in ("qlora", "tuna"):
            status, body, _ = self.request("POST", "/api/runs", self.config(dataset["id"], method=method, system_prompt="Be kind"))
            self.assertEqual(status, 409)
            self.assertIn("CUDA", body["error"])

    def test_invalid_run_configuration_rejected(self):
        dataset = self.upload()
        for changes in ({"method": "magic"}, {"batch_size": True}, {"epochs": -5}, {"learning_rate": "fast"}, {"max_steps": -1}, {"seed": -1}, {"method": "tuna", "system_prompt": ""}, {"offline": "false"}, {"gradient_checkpointing": "yes"}):
            with self.subTest(changes=changes):
                status, _, _ = self.request("POST", "/api/runs", self.config(dataset["id"], **changes))
                self.assertEqual(status, 400)

    def test_real_process_lifecycle_metrics_and_adapter_export(self):
        # This isolated fixture tests subprocess transport, not ML training.
        (self.root / "trainer.py").write_text('''import json, pathlib, sys
output = pathlib.Path(sys.argv[sys.argv.index("--output") + 1])
(output / "adapter_model.safetensors").write_bytes(b"test-fixture")
print(json.dumps({"type":"metric","step":1,"loss":0.5}))
print(json.dumps({"type":"complete","message":"Fixture complete","elapsed_seconds":1.5,"dataset_sha256":"fixture-hash"}))
''', encoding="utf-8")
        dataset = self.upload()
        self.ready()
        status, body, _ = self.request("POST", "/api/runs", self.config(dataset["id"]))
        self.assertEqual(status, 201)
        identifier = body["run"]["id"]
        run = self.wait_run(identifier, "completed")
        self.assertEqual(run["metrics"][0]["loss"], 0.5)
        self.assertEqual(run["summary"]["elapsed_seconds"], 1.5)
        self.assertEqual(run["summary"]["dataset_sha256"], "fixture-hash")
        self.assertTrue(run["artifact_available"])
        self.assertTrue(Path(run["config"]["dataset_path"]).is_absolute())
        self.assertEqual(run["config"]["seed"], 42)
        status, contents, _ = self.request("GET", f"/api/runs/{identifier}/export", raw=True)
        self.assertEqual(status, 200)
        with zipfile.ZipFile(io.BytesIO(contents)) as archive:
            self.assertEqual(archive.namelist(), ["adapter_model.safetensors"])
        self.assertEqual(Studio(self.root, probe=False).get_run(identifier)["status"], "completed")

    def test_exit_without_completion_is_failure(self):
        (self.root / "trainer.py").write_text('print("Exited early")', encoding="utf-8")
        dataset = self.upload()
        self.ready()
        run = self.studio.start_run(self.config(dataset["id"]))
        failed = self.wait_run(run["id"], "failed")
        self.assertIn("without reporting", failed["error"])

    def test_stop_active_run_and_prevent_concurrent_start_or_delete(self):
        (self.root / "trainer.py").write_text('import time\ntime.sleep(60)', encoding="utf-8")
        dataset = self.upload()
        self.ready()
        run = self.studio.start_run(self.config(dataset["id"]))
        self.assertEqual(self.request("POST", "/api/runs", self.config(dataset["id"]))[0], 409)
        self.assertEqual(self.request("DELETE", "/api/datasets/" + dataset["id"])[0], 409)
        status, _, _ = self.request("POST", f'/api/runs/{run["id"]}/stop', {})
        self.assertEqual(status, 200)
        self.wait_run(run["id"], "stopped")
        self.assertEqual(self.request("GET", f'/api/runs/{run["id"]}/export')[0], 409)

    def test_interrupted_run_recovers_as_failed(self):
        identifier = "a" * 32
        atomic_json(self.studio.runs_dir / identifier / "state.json", {"id": identifier, "status": "running", "created_at": "2026-01-01", "config": {}})
        recovered = Studio(self.root, probe=False).get_run(identifier)
        self.assertEqual(recovered["status"], "failed")
        self.assertIn("interrupted", recovered["error"])

    def test_settings_preserve_offline_mode(self):
        self.assertFalse(self.request("GET", "/api/settings")[1]["settings"]["offline"])
        self.assertEqual(self.request("PUT", "/api/settings", {"offline": True})[0], 200)
        self.assertEqual(self.request("PUT", "/api/settings", {"default_model": "local/model", "theme": "dark"})[0], 200)
        self.assertTrue(self.studio.settings()["offline"])
        self.assertTrue(self.studio.get_status()["offline"])
        self.assertEqual(self.request("PUT", "/api/settings", {"offline": False})[0], 200)
        self.assertFalse(self.studio.settings()["offline"])
        self.assertEqual(self.request("PUT", "/api/settings", {"offline": "yes"})[0], 400)
        self.assertEqual(self.studio.settings()["theme"], "dark")

    def test_model_download_permission_reaches_worker(self):
        (self.root / "trainer.py").write_text('''import json, os
assert os.environ["HF_HUB_OFFLINE"] == "0"
assert os.environ["TRANSFORMERS_OFFLINE"] == "0"
print(json.dumps({"type":"complete","message":"Online model loading permitted"}))
''', encoding="utf-8")
        dataset = self.upload()
        self.ready()
        run = self.studio.start_run(self.config(dataset["id"], offline=False))
        self.assertFalse(run["config"]["offline"])
        self.wait_run(run["id"], "completed")

    def test_external_bind_is_rejected(self):
        with self.assertRaises(ValueError):
            make_server(self.studio, 0, "0.0.0.0")

    def completed_fixture(self):
        identifier="b"*32
        folder=self.studio.runs_dir/identifier
        folder.mkdir()
        (folder/"adapter_model.safetensors").write_bytes(b"adapter-fixture")
        run={"id":identifier,"name":"Fixture","status":"completed","created_at":"2026-01-01","config":{},"logs":[],"metrics":[]}
        self.studio.runs[identifier]=run
        self.studio._save_run(run)
        return identifier

    def wait_gguf(self,identifier,state):
        deadline=time.monotonic()+6
        while time.monotonic()<deadline:
            run=self.studio.get_run(identifier)
            if run.get("gguf",{}).get("status")==state:
                return run
            time.sleep(.03)
        self.fail("GGUF export did not reach "+state)

    def test_gguf_lifecycle_download_companion_and_adapter_separation(self):
        # A protocol fixture: real GGUF tensor validity is checked in the worker.
        (self.root/"gguf_export.py").write_text('''import json,pathlib,sys
run=pathlib.Path(sys.argv[sys.argv.index("--run-dir")+1])
output=run/"gguf";output.mkdir(exist_ok=True)
(output/"model.Q8_0.gguf").write_bytes(b"GGUF"+bytes(20))
(output/"Modelfile").write_text("FROM ./model.Q8_0.gguf")
print(json.dumps({"type":"export_progress","stage":"validating","message":"Checking"}))
print(json.dumps({"type":"export_complete","filename":"model.Q8_0.gguf","size_bytes":24}))
''',encoding="utf-8")
        identifier=self.completed_fixture()
        url=f"/api/runs/{identifier}/gguf"
        self.assertEqual(self.request("GET",url)[0],409)
        self.assertEqual(self.request("POST",url,{"format":"imaginary"})[0],400)
        self.assertEqual(self.request("POST",url,{"format":"q8_0"})[0],201)
        run=self.wait_gguf(identifier,"completed")
        status,body,headers=self.request("GET",url,raw=True)
        self.assertEqual(status,200)
        self.assertTrue(body.startswith(b"GGUF"))
        self.assertIn(".gguf",headers["Content-Disposition"])
        self.assertEqual(self.request("POST",url,{"format":"q8_0"})[1]["run"]["gguf"]["started_at"],run["gguf"]["started_at"])
        status,body,_=self.request("GET",url+"/companion",raw=True)
        self.assertEqual(status,200)
        with zipfile.ZipFile(io.BytesIO(body)) as archive:
            self.assertEqual(archive.namelist(),["Modelfile"])
        _,body,_=self.request("GET",f"/api/runs/{identifier}/export",raw=True)
        with zipfile.ZipFile(io.BytesIO(body)) as archive:
            self.assertFalse(any(name.startswith("gguf/") for name in archive.namelist()))

    def test_gguf_cancellation_and_concurrent_work(self):
        (self.root/"gguf_export.py").write_text("import time\ntime.sleep(60)",encoding="utf-8")
        identifier=self.completed_fixture()
        url=f"/api/runs/{identifier}/gguf"
        self.assertEqual(self.request("POST",url,{})[0],201)
        self.assertEqual(self.request("POST",url,{})[0],201)
        dataset=self.upload();self.ready()
        self.assertEqual(self.request("POST","/api/runs",self.config(dataset["id"]))[0],409)
        self.assertEqual(self.request("POST",url+"/stop",{})[0],200)
        self.assertEqual(self.wait_gguf(identifier,"stopped")["status"],"completed")
        self.assertEqual(self.request("GET",url)[0],409)

    def test_gguf_incomplete_worker_cannot_publish_a_download(self):
        (self.root/"gguf_export.py").write_text('print("No GGUF was created")',encoding="utf-8")
        identifier=self.completed_fixture()
        url=f"/api/runs/{identifier}/gguf"
        self.request("POST",url,{})
        self.wait_gguf(identifier,"failed")
        self.assertEqual(self.request("GET",url)[0],409)


if __name__ == "__main__":
    unittest.main()
