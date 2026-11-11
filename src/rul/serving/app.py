"""SageMaker inference contract: GET /ping and POST /invocations on port 8080.

SageMaker starts the container with the argument ``serve`` and extracts the model
artifact (model.tar.gz with model.pt and metadata.json) into /opt/ml/model.

POST /invocations accepts

* ``application/json``: ``{"records": [{"unit", "cycle", "op1".."op3", "s2".."s21"}, ...],
  "adaptation": "auto" | "none" | "anchored", "n_conditions": int | null}``
* ``text/csv`` with a header row of the same columns; adaptation and n_conditions can be
  passed as SageMaker CustomAttributes (``adaptation=auto;n_conditions=6``)
"""

from __future__ import annotations

import io
import json
import logging
import os
import time
from typing import Any

import pandas as pd
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse

from .predictor import InputError, Predictor, to_frame

log = logging.getLogger("rul.serving")
MAX_BYTES = 6 * 1024 * 1024   # SageMaker real-time payload limit


def _custom_attributes(header: str | None) -> dict[str, str]:
    out = {}
    for part in (header or "").split(";"):
        if "=" in part:
            k, v = part.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def create_app(model_dir: str | None = None) -> FastAPI:
    app = FastAPI(title="shift-aware-rul inference", version="0.1.0")
    app.state.model_dir = model_dir or os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
    app.state.predictor = None
    app.state.load_error = None

    def predictor() -> Predictor | None:
        if app.state.predictor is None and app.state.load_error is None:
            try:
                app.state.predictor = Predictor.from_dir(app.state.model_dir)
            except Exception as exc:  # missing or corrupt artifact: /ping reports it
                app.state.load_error = f"{type(exc).__name__}: {exc}"
                log.error("model load failed: %s", app.state.load_error)
        return app.state.predictor

    @app.get("/ping")
    def ping() -> Response:
        p = predictor()
        if p is None:
            return JSONResponse({"status": "unavailable", "error": app.state.load_error}, status_code=503)
        return JSONResponse({"status": "ok", "model": p.meta["created_at"], "source": p.meta["source"]})

    @app.get("/execution-parameters")
    def execution_parameters() -> dict[str, Any]:
        # Read by SageMaker Batch Transform.
        return {"MaxConcurrentTransforms": 1, "BatchStrategy": "MULTI_RECORD", "MaxPayloadInMB": 6}

    @app.post("/invocations")
    async def invocations(request: Request) -> Response:
        start = time.perf_counter()
        p = predictor()
        if p is None:
            return JSONResponse({"error": "model not loaded", "detail": app.state.load_error}, status_code=503)
        body = await request.body()
        if len(body) > MAX_BYTES:
            return JSONResponse({"error": "payload too large"}, status_code=413)
        ctype = request.headers.get("content-type", "application/json").split(";")[0].strip().lower()
        attrs = _custom_attributes(request.headers.get("x-amzn-sagemaker-custom-attributes"))
        try:
            if ctype == "application/json":
                payload = json.loads(body)
                if not isinstance(payload, dict):
                    raise InputError("JSON body must be an object with 'records'")
                df = to_frame(payload.get("records"))
                adaptation = payload.get("adaptation", attrs.get("adaptation", "auto"))
                k = payload.get("n_conditions", attrs.get("n_conditions"))
            elif ctype == "text/csv":
                df = to_frame(pd.read_csv(io.BytesIO(body)).to_dict("records"))
                adaptation, k = attrs.get("adaptation", "auto"), attrs.get("n_conditions")
            else:
                return JSONResponse({"error": f"unsupported content type {ctype!r}"}, status_code=415)
            result = p.predict(df, adaptation=adaptation, n_conditions=int(k) if k else None)
        except (InputError, json.JSONDecodeError, pd.errors.ParserError, ValueError) as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        log.info(json.dumps({"event": "invocation", "engines": len(result["predictions"]),
                             "adaptation": result["adaptation"], "drift_level": result["drift"]["level"],
                             "max_psi_settings": result["drift"]["max_psi_settings"],
                             "latency_ms": round((time.perf_counter() - start) * 1000, 1)}))
        return JSONResponse(result)

    return app
