import asyncio
import os
import pathlib
from typing import Dict, List, Optional

from flask import Flask, jsonify, request, send_file, send_from_directory

from cictify_core import CascadeOrchestrator, OCRIngestion
from cictify_core.config import CONFIG


BASE_DIR = pathlib.Path(__file__).parent.resolve()
GUI_DIR = BASE_DIR / "gui"

app = Flask(__name__, static_folder=None)
loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)

orchestrator: Optional[CascadeOrchestrator] = None
ocr_ingestion: Optional[OCRIngestion] = None
chat_memory: List[Dict[str, str]] = []


def _ensure_ready() -> None:
    global orchestrator, ocr_ingestion
    if orchestrator is not None and ocr_ingestion is not None:
        return

    orchestrator = CascadeOrchestrator()
    orchestrator.initialize()

    # Reuse orchestrator stores to avoid duplicate embedding model load.
    ocr_ingestion = OCRIngestion(orchestrator.rag, floorplan_store=orchestrator.floorplans)


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

    data = request.get_json(force=True, silent=True)
    if not data:
        return jsonify({"reply": "Invalid request", "model": "error"}), 400

    message = data.get("message", "").strip()
    if not message:
        return jsonify({"reply": "Empty message", "model": "error"}), 400

    try:
        _ensure_ready()
        print(f"[Chat] User: {message}", flush=True)

        chat_memory.append({"role": "user", "content": message})
        chat_memory = chat_memory[-CONFIG.max_memory_messages:]

        result, status = loop.run_until_complete(orchestrator.respond(message, chat_memory))
        print(f"[Chat] Route={result.route} Status={status}", flush=True)

        chat_memory.append({"role": "assistant", "content": result.reply})

        return jsonify(
            {
                "reply": result.reply.replace("\n", "<br>"),
                "model": result.route,
                "context": result.context,
                "status": status,
            }
        )
    except Exception:
        return jsonify(
            {
                "reply": "An internal error occurred. Please try again.",
                "model": "error",
            }
        ), 500


@app.route("/api/ingest/image", methods=["POST"])
def ingest_image():
    try:
        _ensure_ready()
        if "file" not in request.files:
            return jsonify({"status": "error", "message": "Missing image file"}), 400

        uploaded = request.files["file"]
        image_bytes = uploaded.read()
        if not image_bytes:
            return jsonify({"status": "error", "message": "Empty file"}), 400

        extracted = loop.run_until_complete(ocr_ingestion.ingest_image(image_bytes))
        if not extracted:
            return jsonify({"status": "error", "message": "OCR extraction failed"}), 500

        return jsonify({"status": "success", "ocr_text": extracted})
    except Exception:
        return jsonify({"status": "error", "message": "OCR endpoint failed"}), 500


@app.route("/api/ingest/floorplan", methods=["POST"])
def ingest_floorplan():
    try:
        _ensure_ready()
        if "file" not in request.files:
            return jsonify({"status": "error", "message": "Missing floorplan image file"}), 400

        uploaded = request.files["file"]
        image_bytes = uploaded.read()
        if not image_bytes:
            return jsonify({"status": "error", "message": "Empty floorplan file"}), 400

        title = (request.form.get("title") or uploaded.filename or "Floorplan").strip()
        building = (request.form.get("building") or "Unknown Building").strip()
        floor = (request.form.get("floor") or "Unknown Floor").strip()

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

        return jsonify({"status": "success", "record": record})
    except Exception:
        return jsonify({"status": "error", "message": "Floorplan endpoint failed"}), 500


@app.route("/api/health", methods=["GET"])
def health_check():
    return jsonify(
        {
            "status": "online",
            "system": "CICTify",
            "architecture": "General -> RAG -> Web (prewarmed)",
            "model_loaded": orchestrator is not None,
            "memory_size": len(chat_memory),
            "max_memory": CONFIG.max_memory_messages,
        }
    )


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
    app.run(host=CONFIG.host, port=CONFIG.port, debug=False)
