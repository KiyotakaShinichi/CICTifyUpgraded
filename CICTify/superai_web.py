import asyncio
import os
import pathlib
import time
import traceback
import uuid
from typing import Dict, List, Optional

from flask import Flask, jsonify, request, send_file, send_from_directory

from cictify_core import CascadeOrchestrator, OCRIngestion
from cictify_core.config import CONFIG, groq_api_key


BASE_DIR = pathlib.Path(__file__).parent.resolve()
GUI_DIR = BASE_DIR / "gui"

app = Flask(__name__, static_folder=None)
loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)

orchestrator: Optional[CascadeOrchestrator] = None
ocr_ingestion: Optional[OCRIngestion] = None
chat_memory: List[Dict[str, str]] = []
startup_ready: bool = False
startup_error: str = ""


def _ensure_ready() -> None:
    global orchestrator, ocr_ingestion
    if orchestrator is not None and ocr_ingestion is not None:
        return

    orchestrator = CascadeOrchestrator()
    orchestrator.initialize()

    # Reuse orchestrator stores to avoid duplicate embedding model load.
    ocr_ingestion = OCRIngestion(
        orchestrator.rag,
        floorplan_store=orchestrator.floorplans,
        spatial_graph_store=orchestrator.spatial_graph,
    )


def _initialize_before_serve() -> None:
    global startup_ready, startup_error
    if startup_ready:
        return

    t0 = time.perf_counter()
    print("[Startup] Initializing orchestrator and stores before serving traffic...", flush=True)
    try:
        _ensure_ready()
        startup_ready = True
        startup_error = ""
        elapsed = time.perf_counter() - t0
        print(f"[Startup] Preload complete in {elapsed:.2f}s", flush=True)
    except Exception as exc:
        startup_ready = False
        startup_error = str(exc)
        raise


@app.route("/")
def index():
    index_path = GUI_DIR / "index.html"
    if index_path.exists():
        return send_file(str(index_path))
    return "CICTify interface not found", 404


@app.route("/favicon.ico")
def favicon():
    favicon_path = GUI_DIR / "images" / "CICTify_ChatLogo.png"
    if favicon_path.exists():
        return send_file(str(favicon_path), mimetype="image/png")
    return "", 204


@app.route("/images/<path:filename>")
def serve_images(filename):
    return send_from_directory(str(GUI_DIR / "images"), filename)


@app.route("/<path:filepath>")
def serve_file(filepath):
    file_path = GUI_DIR / filepath
    if file_path.exists() and file_path.is_file():
        return send_from_directory(str(GUI_DIR), filepath)
    return "File not found", 404


@app.route("/chat", methods=["POST"])
@app.route("/api/chat", methods=["POST"])
def chat_endpoint():
    global chat_memory
    req_id = uuid.uuid4().hex[:8]
    t0 = time.perf_counter()

    data = request.get_json(force=True, silent=True)
    if not data:
        print(f"[API][{req_id}] /api/chat -> 400 invalid-json", flush=True)
        return jsonify({"reply": "Invalid request", "model": "error"}), 400

    message = data.get("message", "").strip()
    if not message:
        print(f"[API][{req_id}] /api/chat -> 400 empty-message", flush=True)
        return jsonify({"reply": "Empty message", "model": "error"}), 400

    try:
        _ensure_ready()
        print(f"[API][{req_id}] User: {message}", flush=True)

        chat_memory.append({"role": "user", "content": message})
        chat_memory = chat_memory[-CONFIG.max_memory_messages:]

        result, status = loop.run_until_complete(orchestrator.respond(message, chat_memory))
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        print(
            f"[API][{req_id}] Route={result.route} Status={status} HTTP=200 ElapsedMs={elapsed_ms}",
            flush=True,
        )

        chat_memory.append({"role": "assistant", "content": result.reply})

        return jsonify(
            {
                "reply": result.reply,
                "model": result.route,
                "context": result.context,
                "status": status,
                "request_id": req_id,
            }
        )
    except Exception as exc:
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        print(
            f"[API][{req_id}] ERROR HTTP=500 ElapsedMs={elapsed_ms} Type={type(exc).__name__} Message={exc}",
            flush=True,
        )
        print(traceback.format_exc(), flush=True)
        return jsonify(
            {
                "reply": "An internal error occurred. Please try again.",
                "model": "error",
                "request_id": req_id,
            }
        ), 500


@app.route("/api/ingest/image", methods=["POST"])
def ingest_image():
    try:
        _ensure_ready()
        if not groq_api_key():
            return jsonify({"status": "error", "message": "GROQ_API_KEY is missing. Set it in CICTify/.env to enable vision ingestion."}), 503
        if "file" not in request.files:
            return jsonify({"status": "error", "message": "Missing image file"}), 400

        uploaded = request.files["file"]
        image_bytes = uploaded.read()
        if not image_bytes:
            return jsonify({"status": "error", "message": "Empty file"}), 400

        title = (request.form.get("title") or uploaded.filename or "Image").strip()
        building = (request.form.get("building") or "Unknown Building").strip()
        floor = (request.form.get("floor") or "Unknown Floor").strip()

        result = loop.run_until_complete(
            ocr_ingestion.ingest_image_auto(
                image_bytes,
                title=title,
                building=building,
                floor=floor,
                source_file=uploaded.filename or "uploaded_image",
            )
        )
        image_type = result.get("image_type", {})

        if image_type.get("type") == "floor_plan" and not result.get("record"):
            return jsonify({"status": "error", "message": "Floorplan detected but ingestion failed", "image_type": image_type}), 500

        return jsonify(
            {
                "status": "success",
                "image_type": image_type,
                "record": result.get("record"),
                "ocr_text": result.get("ocr_text"),
                "status_update": result.get("status_update"),
            }
        )
    except Exception:
        return jsonify({"status": "error", "message": "OCR endpoint failed"}), 500


@app.route("/api/ingest/floorplan", methods=["POST"])
def ingest_floorplan():
    try:
        _ensure_ready()
        if not groq_api_key():
            return jsonify({"status": "error", "message": "GROQ_API_KEY is missing. Set it in CICTify/.env to enable floorplan ingestion."}), 503
        if "file" not in request.files:
            return jsonify({"status": "error", "message": "Missing floorplan image file"}), 400

        uploaded = request.files["file"]
        image_bytes = uploaded.read()
        if not image_bytes:
            return jsonify({"status": "error", "message": "Empty floorplan file"}), 400

        title = (request.form.get("title") or uploaded.filename or "Floorplan").strip()
        building = (request.form.get("building") or "Unknown Building").strip()
        floor = (request.form.get("floor") or "Unknown Floor").strip()

        image_type = loop.run_until_complete(ocr_ingestion.classify_image(image_bytes))
        if image_type.get("type") != "floor_plan" and float(image_type.get("confidence", 0.0)) >= 0.6:
            return jsonify(
                {
                    "status": "error",
                    "message": "Uploaded image is likely not a floor plan",
                    "image_type": image_type,
                }
            ), 400

        record = loop.run_until_complete(
            ocr_ingestion.ingest_floorplan(
                image_bytes,
                title=title,
                building=building,
                floor=floor,
                source_file=uploaded.filename or "uploaded_floorplan",
            )
        )
        if not record:
            return jsonify({"status": "error", "message": "Floorplan OCR ingestion failed"}), 500

        return jsonify({"status": "success", "image_type": image_type, "record": record})
    except Exception:
        return jsonify({"status": "error", "message": "Floorplan endpoint failed"}), 500


@app.route("/api/health", methods=["GET"])
def health_check():
    key_available = bool(groq_api_key())
    spatial_stats = orchestrator.spatial_graph.stats() if orchestrator is not None else {"nodes": 0, "edges": 0}
    return jsonify(
        {
            "status": "online" if startup_ready else "starting",
            "system": "CICTify",
            "architecture": "General -> Spatial RAG -> RAG -> Web (prewarmed)",
            "model_loaded": startup_ready and orchestrator is not None,
            "memory_size": len(chat_memory),
            "max_memory": CONFIG.max_memory_messages,
            "startup_ready": startup_ready,
            "startup_error": startup_error,
            "dependencies": {
                "groq_api_key": key_available,
            },
            "spatial_graph": spatial_stats,
        }
    )


@app.route("/api/route", methods=["POST"])
def route_endpoint():
    global chat_memory
    try:
        _ensure_ready()
        data = request.get_json(force=True, silent=True) or {}
        start_hint = str(data.get("from") or "").strip()
        target_hint = str(data.get("to") or "").strip()
        start_id = str(data.get("from_id") or "").strip()
        target_id = str(data.get("to_id") or "").strip()
        start_label = str(data.get("from_label") or start_hint or start_id).strip()
        target_label = str(data.get("to_label") or target_hint or target_id).strip()
        algorithm = str(data.get("algorithm") or "astar").strip().lower()

        if not (start_id and target_id) and not (start_hint and target_hint):
            return jsonify({"status": "error", "message": "Provide either from/to ids or from/to names."}), 400

        if algorithm not in {"astar", "dijkstra"}:
            algorithm = "astar"

        if start_id and target_id:
            route = orchestrator.spatial_graph.compute_route_by_ids(start_id, target_id, algorithm=algorithm)
        else:
            route = orchestrator.spatial_graph.compute_route(start_hint, target_hint, algorithm=algorithm)
        if not route:
            return jsonify(
                {
                    "status": "error",
                    "message": "Route not found. Check place names or ingest more floorplan graph data.",
                }
            ), 404

        directions = route.get("directions", [])
        if directions:
            guide_lines = "\n".join(f"{idx}. {step}" for idx, step in enumerate(directions, start=1))
            reply = (
                f"Sure. Here is the best mapped route from {route.get('start', {}).get('name', start_label)} "
                f"to {route.get('target', {}).get('name', target_label)}.\n"
                f"{guide_lines}\n"
                f"Estimated path cost: {route.get('distance', 'n/a')}"
            )
        else:
            path_names = route.get("path_names", [])
            path_text = " -> ".join(path_names) if path_names else "No detailed path available."
            reply = (
                f"I found a route from {start_label} to {target_label}.\n"
                f"Path: {path_text}\n"
                f"Estimated path cost: {route.get('distance', 'n/a')}"
            )

        synthetic_user = f"How do I get from {start_label} to {target_label}?"
        chat_memory.append({"role": "user", "content": synthetic_user})
        chat_memory.append({"role": "assistant", "content": reply})
        chat_memory = chat_memory[-CONFIG.max_memory_messages:]

        return jsonify({"status": "success", "route": route, "reply": reply})
    except Exception as exc:
        print(f"[API][route] ERROR Type={type(exc).__name__} Message={exc}", flush=True)
        print(traceback.format_exc(), flush=True)
        return jsonify({"status": "error", "message": "Route endpoint failed"}), 500


@app.route("/api/route/options", methods=["GET"])
def route_options_endpoint():
    try:
        _ensure_ready()
        options = orchestrator.spatial_graph.route_options()
        return jsonify({"status": "success", "options": options})
    except Exception as exc:
        print(f"[API][route-options] ERROR Type={type(exc).__name__} Message={exc}", flush=True)
        print(traceback.format_exc(), flush=True)
        return jsonify({"status": "error", "message": "Route options endpoint failed"}), 500


@app.route("/api/reset", methods=["POST"])
def reset_conversation():
    global chat_memory
    chat_memory = []
    return jsonify({"status": "success", "message": "Conversation memory cleared"})


if __name__ == "__main__":
    print(f"[Startup] Running on http://127.0.0.1:{CONFIG.port}", flush=True)
    print("[Startup] Route trace logs enabled: [Chat], [Query], [Router], [Cascade]", flush=True)
    print(f"[Startup] Router model: {CONFIG.routing_model}", flush=True)
    print(f"[Startup] Answer model(s): {', '.join(CONFIG.chat_models)}", flush=True)
    print(f"[Startup] Retrieval: FAISS similarity (k={CONFIG.retrieval_k})", flush=True)
    print(
        "[Startup] Hyperparameters: "
        f"CHUNK_SIZE={CONFIG.CHUNK_SIZE}, "
        f"CHUNK_OVERLAP={CONFIG.CHUNK_OVERLAP}, "
        f"MAX_TOKENS={CONFIG.MAX_TOKENS}, "
        f"TEMPERATURE={CONFIG.TEMPERATURE}, "
        f"RETRIEVAL_K={CONFIG.RETRIEVAL_K}, "
        f"RAG_RERANK_TOP_N={CONFIG.RAG_RERANK_TOP_N}, "
        f"RAG_MIN_TERM_OVERLAP={CONFIG.RAG_MIN_TERM_OVERLAP}, "
        f"MEMORY_TOP_K={CONFIG.MEMORY_TOP_K}, "
        f"MEMORY_MIN_SCORE={CONFIG.MEMORY_MIN_SCORE}",
        flush=True,
    )
    print(f"[Startup] Web mode: prewarmed cache (live fallback={CONFIG.web_live_fallback})", flush=True)
    print(f"[Startup] GROQ_API_KEY configured: {bool(groq_api_key())}", flush=True)
    try:
        _initialize_before_serve()
    except Exception as exc:
        print(f"[Startup] Preload failed. Server will not start: {exc}", flush=True)
        raise SystemExit(1)
    app.run(host=CONFIG.host, port=CONFIG.port, debug=False)
