# CICTify

CICTify is an AI assistant for BulSU CICT that combines:

- Retrieval-augmented QA over curated university documents
- Spatial graph reasoning for buildings, halls, and navigation-style questions
- OCR + computer vision ingestion for floor plans and room-status images

## Quick Start (Windows)

1. Create and activate a virtual environment.

```powershell
cd CICTify
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies.

```powershell
pip install -r requirements.txt
playwright install
```

3. Create `.env` in `CICTify/` and set at least:

```env
GROQ_API_KEY=your_key_here
```

Optional vision model override:

```env
VISION_MODEL=meta-llama/llama-4-scout-17b-16e-instruct,meta-llama/llama-4-maverick-17b-128e-instruct
```

4. Run the app.

```powershell
python superai_web.py
```

Then open `http://127.0.0.1:5000`.

## Deployment

Recommended production path:

1. Build the app as a Docker image using the root [Dockerfile](../Dockerfile).
2. Deploy the image on Render using the root [render.yaml](../render.yaml).
3. Keep `GROQ_API_KEY` and other secrets as Render environment variables, not in the image.

### Local Docker run

```powershell
cd CICTifyUpgraded
docker build -t cictify .
docker run --rm -p 10000:10000 -e GROQ_API_KEY=your_key_here cictify
```

Then open `http://127.0.0.1:10000`.

### Why this pipeline

- Docker makes the runtime reproducible across machines.
- Render is the quickest reliable host for a Python AI app like CICTify.
- Gunicorn keeps the Flask app production-ready without a framework rewrite.
- Kubernetes is unnecessary at this stage unless you need multi-service autoscaling.

## Key API Endpoints

- `POST /api/chat`: main chat endpoint
- `POST /api/ingest/image`: auto-classify image and ingest (floor plan or room-status image)
- `POST /api/ingest/floorplan`: strict floor-plan ingestion path
- `GET /api/health`: health and spatial graph stats
- `POST /api/reset`: clear conversation memory

## How OCR + CV Works For Floor Plans

### 1) Ingestion entry point

The web app accepts uploaded images through `POST /api/ingest/image` and `POST /api/ingest/floorplan` in [superai_web.py](superai_web.py).

Both routes pass image bytes to `OCRIngestion` from [cictify_core/ocr_ingestion.py](cictify_core/ocr_ingestion.py).

### 2) Vision classification

`OCRIngestion.ingest_image_auto(...)` first calls `classify_image_with_groq(...)` in [vision_reader.py](vision_reader.py) to label the image as:

- `floor_plan`
- `room_photo`
- `document`
- `other`

### 3) OCR text extraction

All image understanding uses `read_image_with_groq(...)` in [vision_reader.py](vision_reader.py), which:

- Encodes image bytes to base64
- Sends a multimodal prompt to Groq Vision models
- Tries model fallbacks if one model is unavailable
- Returns extracted text / reasoning output

### 4) Floor-plan path

If classified as `floor_plan`, `OCRIngestion.ingest_floorplan(...)` does two writes:

- Saves OCR text and metadata to floorplan context store via [cictify_core/floorplan_context.py](cictify_core/floorplan_context.py)
- Attempts graph extraction (`nodes` + `edges`) via `extract_floorplan_graph_with_groq(...)` in [vision_reader.py](vision_reader.py), then merges into spatial graph via [cictify_core/spatial_graph.py](cictify_core/spatial_graph.py)

This is what allows follow-up spatial answers like:

- "where is natividad hall"
- "what is near natividad hall"

### 5) Non-floor-plan path

If image is not a floor plan, CICTify still runs OCR and may extract room status (`has_open_pc`, room metadata) via `extract_room_status_with_groq(...)` in [vision_reader.py](vision_reader.py), then upserts it into the spatial graph.

## Practical Upload Notes

- Use clear, high-resolution map/floor-plan images.
- Provide `building` and `floor` form fields when uploading to improve graph quality.
- OCR from handbook pages can be noisy if the page is image-heavy; multiple uploads or cleaner crops improve results.

## Project Structure (Core)

- [superai_web.py](superai_web.py): Flask server and ingestion routes
- [vision_reader.py](vision_reader.py): Groq vision OCR/classification/graph extraction
- [cictify_core/ocr_ingestion.py](cictify_core/ocr_ingestion.py): ingestion orchestration
- [cictify_core/floorplan_context.py](cictify_core/floorplan_context.py): OCR floor-plan context persistence
- [cictify_core/spatial_graph.py](cictify_core/spatial_graph.py): spatial graph, near/where/route reasoning
- [cictify_core/orchestrator.py](cictify_core/orchestrator.py): cascade routing and response generation
