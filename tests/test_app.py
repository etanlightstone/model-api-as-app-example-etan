"""End-to-end tests for the Model-API-as-App harness.

Run with: ``python -m unittest tests.test_app`` (no pytest dependency).

Uses the FastAPI TestClient and the *real* example models (their trained
artifacts are present), exercising schema inference, the sync endpoint, the
async submit/poll engine (thread backend), and owner gating.
"""

from __future__ import annotations

import os
import tempfile
import time
import unittest
import warnings

warnings.filterwarnings("ignore")

# Configure the harness BEFORE importing app modules.
_TMP = tempfile.mkdtemp(prefix="model_app_test_")
os.environ["MODEL_APP_DATA_DIR"] = _TMP
os.environ["MODEL_APP_TASKS_BACKEND"] = "thread"   # no process spawn in tests
os.environ["MODEL_APP_DEV_OWNER"] = "1"            # treat caller as owner by default
os.environ.setdefault("MODEL_APP_TASKS_POLL_SECONDS", "1")
os.environ.setdefault("MODEL_APP_TASKS_HEARTBEAT_SECONDS", "5")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEATHER = os.path.join(REPO, "example/weather_regressor/model_api.py")
DIABETES = os.path.join(REPO, "example/diabetes_classer/model_api.py")
IMAGE = os.path.join(REPO, "example/image_classifier/model_api.py")
PASSTHROUGH = os.path.join(REPO, "tests/fixtures/passthrough_model.py")


def _make_image_b64(color=(40, 90, 230)):
    import base64
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (48, 48), color).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

from fastapi.testclient import TestClient  # noqa: E402

from core import config as config_mod  # noqa: E402
from core import db, state  # noqa: E402


def _reset_db():
    db.init_db()
    with db.transaction() as conn:
        conn.execute("DELETE FROM app_config")
        conn.execute("DELETE FROM inference_tasks")
    state.reload_from_config()


class SchemaInferenceTests(unittest.TestCase):
    def test_custom_function_schema_both_examples(self):
        from core.adapter import CustomFunctionAdapter
        from core.schema import example_record

        a = CustomFunctionAdapter(WEATHER, "predict")
        a.ensure_warm()
        self.assertEqual(a.slug, "weather-regressor")
        self.assertEqual(a.input_schema.input_names(),
                         ["month", "week_of", "state", "precipitation", "wind_speed", "wind_direction"])
        out_names = [f.name for f in a.input_schema.outputs]
        self.assertEqual(out_names, ["avg_temp", "max_temp", "min_temp"])

        res = a.predict([{"month": "7", "week_of": "28", "state": "Alabama",
                          "precipitation": "0.1", "wind_speed": "5.0", "wind_direction": "20"}])
        self.assertEqual(len(res), 1)
        self.assertIn("avg_temp", res[0])

        d = CustomFunctionAdapter(DIABETES, "predict")
        d.ensure_warm()
        self.assertEqual([f.name for f in d.input_schema.outputs],
                         ["is_diabetic", "probability", "threshold"])
        _ = example_record(d.input_schema)


class AppTests(unittest.TestCase):
    def setUp(self):
        _reset_db()
        self.client = TestClient(__import__("app").app)
        self.client.__enter__()  # triggers lifespan (worker startup)

    def tearDown(self):
        self.client.__exit__(None, None, None)

    def _host_weather(self):
        r = self.client.post("/settings/select", json={
            "source_type": "custom_function", "file_path": WEATHER, "func_name": "predict",
            "display_name": "Weather Regressor",
        })
        self.assertEqual(r.status_code, 200, r.text)
        return r.json()

    # --- not-set-up + health ---
    def test_not_set_up_then_health(self):
        home = self.client.get("/")
        self.assertEqual(home.status_code, 200)
        self.assertIn("not set up yet", home.text)

        h = self.client.get("/health")
        self.assertEqual(h.status_code, 200)
        self.assertFalse(h.json()["configured"])

    # --- settings + sync ---
    def test_select_and_sync_predict(self):
        sel = self._host_weather()
        self.assertEqual(sel["slug"], "weather-regressor")
        self.assertTrue(sel["ready"])

        # Home now renders the endpoints page.
        home = self.client.get("/")
        self.assertIn("weather-regressor", home.text)
        self.assertIn("Playground", home.text)

        # Domino-style envelope.
        r = self.client.post("/models/weather-regressor/latest/model", json={
            "data": {"month": "7", "week_of": "28", "state": "Alabama",
                     "precipitation": "0.1", "wind_speed": "5.0", "wind_direction": "20"}})
        self.assertEqual(r.status_code, 200, r.text)
        body = r.json()
        self.assertIn("result", body)
        self.assertIn("request_id", body)
        self.assertIn("avg_temp", body["result"])

        # Bare record (no envelope) also accepted.
        r2 = self.client.post("/models/weather-regressor/latest/model", json={
            "month": "1", "week_of": "2", "state": "Texas",
            "precipitation": "0.0", "wind_speed": "3.0", "wind_direction": "90"})
        self.assertEqual(r2.status_code, 200, r2.text)

    def test_sync_validation_error(self):
        self._host_weather()
        # Missing required fields → 422.
        r = self.client.post("/models/weather-regressor/latest/model",
                             json={"data": {"month": "7"}})
        self.assertEqual(r.status_code, 422, r.text)

    def test_sync_unconfigured_slug(self):
        self._host_weather()
        r = self.client.post("/models/nope/latest/model", json={"data": {}})
        self.assertEqual(r.status_code, 404)

    # --- async by value ---
    def _poll_until_terminal(self, slug, pid, timeout=30):
        terminal = {"succeeded", "failed", "cancelled", "expired"}
        deadline = time.time() + timeout
        while time.time() < deadline:
            r = self.client.get(f"/api/modelApis/async/v1/{slug}/{pid}")
            self.assertEqual(r.status_code, 200, r.text)
            body = r.json()
            if body["status"] in terminal:
                return body
            time.sleep(0.5)
        self.fail(f"task {pid} did not finish within {timeout}s")

    def test_async_by_value(self):
        self._host_weather()
        sub = self.client.post("/api/modelApis/async/v1/weather-regressor", json={
            "parameters": {"month": "7", "week_of": "28", "state": "Alabama",
                           "precipitation": "0.1", "wind_speed": "5.0", "wind_direction": "20"}})
        self.assertEqual(sub.status_code, 200, sub.text)
        pid = sub.json()["asyncPredictionId"]
        self.assertTrue(pid.startswith("task_"))

        body = self._poll_until_terminal("weather-regressor", pid)
        self.assertEqual(body["status"], "succeeded", body)
        self.assertIn("avg_temp", body["result"])

    def test_async_by_reference(self):
        self._host_weather()
        # Write a small JSONL input file as the by-reference payload.
        ref = os.path.join(_TMP, "weather_in.jsonl")
        import json
        with open(ref, "w") as fh:
            for st in ["Alabama", "Texas", "Ohio"]:
                fh.write(json.dumps({"month": "6", "week_of": "24", "state": st,
                                     "precipitation": "0.2", "wind_speed": "4.0",
                                     "wind_direction": "45"}) + "\n")

        sub = self.client.post("/api/modelApis/async/v1/weather-regressor",
                              json={"parameters": {"input_file": ref}})
        self.assertEqual(sub.status_code, 200, sub.text)
        pid = sub.json()["asyncPredictionId"]
        body = self._poll_until_terminal("weather-regressor", pid)
        self.assertEqual(body["status"], "succeeded", body)
        self.assertEqual(body["result"]["completed_items"], 3)

        # Result stream returns 3 rows.
        rs = self.client.get(f"/api/modelApis/async/v1/weather-regressor/{pid}/result")
        self.assertEqual(rs.status_code, 200)
        lines = [l for l in rs.text.strip().splitlines() if l.strip()]
        self.assertEqual(len(lines), 3)

    def test_async_bad_reference(self):
        self._host_weather()
        sub = self.client.post("/api/modelApis/async/v1/weather-regressor",
                              json={"parameters": {"input_file": "/does/not/exist.jsonl"}})
        self.assertEqual(sub.status_code, 422, sub.text)


class ImageClassifierTests(unittest.TestCase):
    def test_image_schema_flag_and_predict(self):
        from core.adapter import CustomFunctionAdapter

        a = CustomFunctionAdapter(IMAGE, "predict")
        a.ensure_warm()
        # model_app.yaml pins name/slug and marks `image` as an image field.
        self.assertEqual(a.slug, "image-classifier")
        self.assertTrue(a.input_schema.has_image_input())
        img_field = a.input_schema.inputs[0]
        self.assertEqual(img_field.name, "image")
        self.assertTrue(img_field.image)
        # Output fields come from the sidecar.
        self.assertEqual([f.name for f in a.input_schema.outputs], ["label", "probabilities"])

        res = a.predict([{"image": _make_image_b64((20, 60, 220))}])
        self.assertIn(res[0]["label"], ["dark", "bright", "grayscale", "colorful"])

    def test_image_host_renders_file_picker(self):
        _reset_db()
        with TestClient(__import__("app").app) as client:
            r = client.post("/settings/select", json={
                "source_type": "custom_function", "file_path": IMAGE, "func_name": "predict"})
            self.assertEqual(r.status_code, 200, r.text)
            html = client.get("/").text
            self.assertIn('type="file"', html)          # playground file picker
            self.assertIn("image input", html)            # chip
            # Live sync call through the JSON endpoint with a base64 image.
            pred = client.post("/models/image-classifier/latest/model",
                              json={"data": {"image": _make_image_b64()}})
            self.assertEqual(pred.status_code, 200, pred.text)
            self.assertIn("label", pred.json()["result"])


class PassthroughTests(unittest.TestCase):
    def test_schema_marked_passthrough(self):
        from core.adapter import CustomFunctionAdapter

        a = CustomFunctionAdapter(PASSTHROUGH, "predict")
        a.ensure_warm()
        self.assertTrue(a.input_schema.passthrough)
        self.assertEqual(a.input_schema.inputs, [])
        # Arbitrary keys flow straight through to the model.
        res = a.predict([{"anything": 1, "goes": "here"}])
        self.assertEqual(res[0]["num_fields"], 2)
        self.assertEqual(res[0]["received"]["goes"], "here")

    def test_passthrough_endpoints(self):
        _reset_db()
        with TestClient(__import__("app").app) as client:
            r = client.post("/settings/select", json={
                "source_type": "custom_function", "file_path": PASSTHROUGH,
                "func_name": "predict", "display_name": "Anything"})
            self.assertEqual(r.status_code, 200, r.text)
            slug = r.json()["slug"]

            # UI shows the passthrough note + raw-JSON playground, not a field table.
            html = client.get("/").text
            self.assertIn("arbitrary JSON", html)
            self.assertIn('data-raw="1"', html)

            # Sync: any JSON shape is accepted and forwarded (no validation error).
            sync = client.post(f"/models/{slug}/latest/model",
                              json={"data": {"foo": "bar", "n": 3}})
            self.assertEqual(sync.status_code, 200, sync.text)
            self.assertEqual(sync.json()["result"]["received"]["foo"], "bar")

            # Async by-value works the same way.
            sub = client.post(f"/api/modelApis/async/v1/{slug}",
                            json={"parameters": {"x": [1, 2, 3]}})
            self.assertEqual(sub.status_code, 200, sub.text)
            pid = sub.json()["asyncPredictionId"]
            terminal = {"succeeded", "failed", "cancelled", "expired"}
            deadline = time.time() + 30
            body = None
            while time.time() < deadline:
                body = client.get(f"/api/modelApis/async/v1/{slug}/{pid}").json()
                if body["status"] in terminal:
                    break
                time.sleep(0.5)
            self.assertEqual(body["status"], "succeeded", body)
            # The list-valued field was NOT columnar-expanded in passthrough mode.
            self.assertEqual(body["result"]["received"]["x"], [1, 2, 3])


class ProxyPathTests(unittest.TestCase):
    """The app is served under an unknown reverse-proxy prefix that nginx strips.
    URLs must be relative / placeholder-based so nothing assumes the root path."""

    def test_base_href_depth(self):
        from types import SimpleNamespace

        from core import links

        def req(path):
            return SimpleNamespace(scope={"path": path}, url=SimpleNamespace(path=path))

        self.assertEqual(links.base_href(req("/")), "./")
        self.assertEqual(links.base_href(req("/settings")), "./")
        self.assertEqual(links.base_href(req("/a/b")), "../")

    def test_pages_use_relative_assets_and_base_tag(self):
        _reset_db()
        with TestClient(__import__("app").app) as client:
            client.post("/settings/select", json={
                "source_type": "custom_function", "file_path": WEATHER, "func_name": "predict"})
            html = client.get("/").text
            # No root-absolute asset links that would escape the proxy prefix.
            self.assertIn('href="static/style.css"', html)
            self.assertIn('src="static/app.js"', html)
            self.assertNotIn('href="/static/', html)
            self.assertNotIn('src="/static/', html)
            self.assertIn("<base href=", html)
            # Absolute base is a placeholder, filled client-side from document.baseURI.
            self.assertIn("__APP_BASE__/models/weather-regressor/latest/model", html)
            self.assertNotIn("http://testserver/models", html)

    def test_settings_page_relative_assets(self):
        _reset_db()
        with TestClient(__import__("app").app) as client:
            html = client.get("/settings").text
            self.assertIn('src="static/settings.js"', html)
            self.assertNotIn('src="/static/', html)
            self.assertIn("<base href=", html)


class OwnerGatingTests(unittest.TestCase):
    def setUp(self):
        _reset_db()

    def test_identifier_normalization_matches(self):
        from core import identity as identity_mod
        # username (underscore), email local-part (dot), and full email all match.
        self.assertEqual(identity_mod._norm("etan_lightstone"), identity_mod._norm("etan.lightstone"))
        self.assertTrue(identity_mod._norm("Etan-Lightstone") == identity_mod._norm("etanlightstone"))

    def test_non_owner_cannot_select(self):
        # Turn off dev-owner mode and provide an identity header that isn't the owner.
        os.environ["MODEL_APP_DEV_OWNER"] = "0"
        os.environ["DOMINO_PROJECT_OWNER"] = "alice"
        os.environ["MODEL_APP_USER_HEADER"] = "x-test-user"
        import importlib

        from core import settings as settings_mod
        importlib.reload(settings_mod)
        from core import identity as identity_mod
        importlib.reload(identity_mod)
        # Reload modules that captured settings at import.
        import routes.settings as settings_routes
        importlib.reload(settings_routes)
        import app as app_mod
        importlib.reload(app_mod)

        try:
            with TestClient(app_mod.app) as client:
                r = client.post("/settings/select",
                               headers={"x-test-user": "bob"},
                               json={"source_type": "custom_function",
                                     "file_path": WEATHER, "func_name": "predict"})
                self.assertEqual(r.status_code, 403, r.text)

                # The owner (alice) is allowed.
                r2 = client.post("/settings/select",
                                headers={"x-test-user": "alice"},
                                json={"source_type": "custom_function",
                                      "file_path": WEATHER, "func_name": "predict"})
                self.assertEqual(r2.status_code, 200, r2.text)
        finally:
            os.environ["MODEL_APP_DEV_OWNER"] = "1"
            os.environ.pop("MODEL_APP_USER_HEADER", None)
            importlib.reload(settings_mod)
            importlib.reload(identity_mod)


if __name__ == "__main__":
    unittest.main(verbosity=2)
