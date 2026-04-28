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
PLAN_ASSETS_DIR = GUI_DIR / "plan_assets"
PDF_DIR = BASE_DIR / "knowledge_base" / "pdfs"
CAMPU_MAP_PDF = PDF_DIR / "BulSUMap.pdf"
CAMPUS_PLAN_ASSET_NAME = "bulsu-main-campus-campus.png"
MIN_CAMPUS_PLAN_WIDTH = 2600

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


def _slug(text: str) -> str:
    import re
    return re.sub(r"[^a-z0-9]+", "-", str(text or "").strip().lower()).strip("-")


def _safe_ext(filename: str) -> str:
    ext = pathlib.Path(str(filename or "")).suffix.lower()
    return ext if ext in {".png", ".jpg", ".jpeg", ".webp"} else ".png"


def _ensure_campus_plan_asset_from_pdf() -> Optional[str]:
    PLAN_ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    target = PLAN_ASSETS_DIR / CAMPUS_PLAN_ASSET_NAME
    if target.exists() and target.stat().st_size > 0:
        try:
            from PIL import Image

            with Image.open(str(target)) as existing:
                if int(existing.width or 0) >= MIN_CAMPUS_PLAN_WIDTH:
                    return f"/plan_assets/{CAMPUS_PLAN_ASSET_NAME}"
        except Exception:
            return f"/plan_assets/{CAMPUS_PLAN_ASSET_NAME}"

    if not CAMPU_MAP_PDF.exists():
        return None

    try:
        import pypdfium2 as pdfium  # type: ignore

        pdf = pdfium.PdfDocument(str(CAMPU_MAP_PDF))
        if len(pdf) < 1:
            return None

        page = pdf[0]
        # Higher DPI export keeps the map readable when users resize/zoom the route panel.
        bitmap = page.render(scale=4.0)
        pil_image = bitmap.to_pil()
        # Trim white page margins so plotted coordinates align to the visible map area.
        try:
            from PIL import ImageChops

            rgb = pil_image.convert("RGB")
            bg = rgb.copy()
            bg.paste((255, 255, 255), [0, 0, rgb.size[0], rgb.size[1]])
            diff = ImageChops.difference(rgb, bg)
            bbox = diff.getbbox()
            if bbox:
                pil_image = rgb.crop(bbox)
        except Exception:
            pass
        pil_image.save(str(target), format="PNG")
        return f"/plan_assets/{CAMPUS_PLAN_ASSET_NAME}"
    except Exception as exc:
        print(f"[Startup] Campus plan generation skipped: {exc}", flush=True)
        return None


def _save_plan_asset(*, image_bytes: bytes, filename: str, building: str, floor: str) -> str:
    PLAN_ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    ext = _safe_ext(filename)
    build_slug = _slug(building) or "unknown-building"
    floor_slug = _slug(floor) or "unknown-floor"
    out_name = f"{build_slug}-{floor_slug}{ext}"
    out_path = PLAN_ASSETS_DIR / out_name
    out_path.write_bytes(image_bytes)
    return f"/plan_assets/{out_name}"


def _resolve_plan_background_url(route: Dict) -> Optional[str]:
    campus_asset_url = _ensure_campus_plan_asset_from_pdf()
    points = route.get("points", []) if isinstance(route, dict) else []
    floors = [str(p.get("floor") or "").strip().lower() for p in points]
    campus_mode = any(f in {"campus", "campus map"} for f in floors)

    if not PLAN_ASSETS_DIR.exists():
        return campus_asset_url if campus_mode else None

    files = [p for p in PLAN_ASSETS_DIR.iterdir() if p.is_file()]
    if not files:
        return campus_asset_url if campus_mode else None

    if not points:
        return campus_asset_url if campus_mode else None

    buildings = [str(p.get("building") or "").strip() for p in points]
    uniq_buildings = sorted({b for b in buildings if b})

    # Cross-building route: prefer campus-level plan background.
    if len(uniq_buildings) > 1:
        if campus_asset_url:
            return campus_asset_url
        for p in files:
            stem = p.stem.lower()
            if any(tok in stem for tok in ["campus", "bulsu-main-campus", "main-campus"]):
                return f"/plan_assets/{p.name}"

    # Same-building route: prefer exact building-floor plan.
    candidates = []
    for b, f in zip(buildings, floors):
        bs = _slug(b)
        fs = _slug(f)
        if bs and fs:
            candidates.append((bs, fs))

    for bs, fs in candidates:
        for p in files:
            stem = p.stem.lower()
            if bs in stem and fs in stem:
                return f"/plan_assets/{p.name}"

    # Fallback to first matching building-only plan.
    for b in buildings:
        bs = _slug(b)
        if not bs:
            continue
        for p in files:
            if bs in p.stem.lower():
                return f"/plan_assets/{p.name}"

    return campus_asset_url if campus_mode else None


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


@app.route("/plan_assets/<path:filename>")
def serve_plan_assets(filename):
    return send_from_directory(str(PLAN_ASSETS_DIR), filename)


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

    # Don't block the worker while the application is prewarming.
    if not startup_ready:
        print(f"[API][{req_id}] Service starting, rejecting request", flush=True)
        return jsonify({"reply": "Service is starting. Please try again shortly.", "model": "starting"}), 503

    try:
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
        if not startup_ready:
            return jsonify({"status": "error", "message": "Service starting. Try again later."}), 503
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
        plan_asset_url = _save_plan_asset(
            image_bytes=image_bytes,
            filename=uploaded.filename or "uploaded_image.png",
            building=building,
            floor=floor,
        )

        result = loop.run_until_complete(
            ocr_ingestion.ingest_image_auto(
                image_bytes,
                title=title,
                building=building,
                floor=floor,
                source_file=plan_asset_url,
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
                "plan_asset_url": plan_asset_url,
            }
        )
    except Exception:
        return jsonify({"status": "error", "message": "OCR endpoint failed"}), 500


@app.route("/api/ingest/floorplan", methods=["POST"])
def ingest_floorplan():
    try:
        if not startup_ready:
            return jsonify({"status": "error", "message": "Service starting. Try again later."}), 503
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
        plan_asset_url = _save_plan_asset(
            image_bytes=image_bytes,
            filename=uploaded.filename or "uploaded_floorplan.png",
            building=building,
            floor=floor,
        )

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
                source_file=plan_asset_url,
            )
        )
        if not record:
            return jsonify({"status": "error", "message": "Floorplan OCR ingestion failed"}), 500

        return jsonify({"status": "success", "image_type": image_type, "record": record, "plan_asset_url": plan_asset_url})
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
        if not startup_ready:
            return jsonify({"status": "error", "message": "Service starting. Try again later."}), 503
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

        # Safety guardrail: limit routing to same domain only.
        # Allowed: CICT<->CICT or Campus<->Campus. Disallow mixed-domain routes.
        place_type_by_id: Dict[str, str] = {}
        try:
            for opt in orchestrator.spatial_graph.route_options():
                oid = str(opt.get("id") or "").strip()
                if oid:
                    place_type_by_id[oid] = str(opt.get("place_type") or "").strip()
        except Exception:
            place_type_by_id = {}

        if start_id and target_id:
            src_type = place_type_by_id.get(start_id, "")
            dst_type = place_type_by_id.get(target_id, "")
            if src_type and dst_type and src_type != dst_type:
                return jsonify(
                    {
                        "status": "error",
                        "message": (
                            "Mixed-domain routing is disabled for now. "
                            "Please choose CICT-to-CICT or Campus-to-Campus only."
                        ),
                    }
                ), 400

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
        if not directions:
            path_names = [str(name).strip() for name in route.get("path_names", []) if str(name).strip()]
            if path_names:
                synthesized = [f"Start at {path_names[0]}."]
                for idx in range(1, len(path_names)):
                    synthesized.append(f"Proceed to {path_names[idx]}.")
                if len(path_names) > 1:
                    synthesized.append(f"You have arrived at {path_names[-1]}.")
                directions = synthesized
                route["directions"] = synthesized
                route["directions_text"] = "\n".join(
                    f"{idx}. {step}" for idx, step in enumerate(synthesized, start=1)
                )

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

        plan_background_url = _resolve_plan_background_url(route)
        return jsonify({"status": "success", "route": route, "reply": reply, "plan_background_url": plan_background_url})
    except Exception as exc:
        print(f"[API][route] ERROR Type={type(exc).__name__} Message={exc}", flush=True)
        print(traceback.format_exc(), flush=True)
        return jsonify({"status": "error", "message": "Route endpoint failed"}), 500


@app.route("/api/route/options", methods=["GET"])
def route_options_endpoint():
    try:
        if not startup_ready:
            return jsonify({"status": "error", "message": "Service starting. Try again later."}), 503
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


# When running under a WSGI server (e.g., Gunicorn on Render), start
# initialization in a background thread so health checks and lightweight
# endpoints do not block while heavy prewarm tasks run.
if __name__ != "__main__":
    import threading
    import time

    def _bg_init():
        # slight delay so the runtime is stable
        time.sleep(0.5)
        try:
            _initialize_before_serve()
        except Exception as exc:
            print(f"[Startup][bg] Initialization failed: {exc}", flush=True)

    threading.Thread(target=_bg_init, daemon=True).start()


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
