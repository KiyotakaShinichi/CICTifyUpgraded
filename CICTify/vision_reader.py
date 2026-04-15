# vision_reader.py
import base64
import json
import os
import re
from typing import Dict, Optional

import aiohttp

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
VISION_MODEL = "meta-llama/llama-4-maverick-17b-128e-instruct"


def _groq_api_key() -> str:
    return os.getenv("GROQ_API_KEY", "").strip()

async def read_image_with_groq(image_bytes: bytes, question: str = "Read all text and describe this image.") -> Optional[str]:
    """Use Groq Vision to OCR and understand any image (e.g., floor plan, diagram)."""
    try:
        api_key = _groq_api_key()
        if not api_key:
            return None
        img_b64 = base64.b64encode(image_bytes).decode("utf-8")

        payload = {
            "model": VISION_MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": "You are an advanced OCR and visual reasoning assistant. "
                               "Extract text, room labels, and layout descriptions from images and floor plans."
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question},
                        {"type": "image_url", "image_url": f"data:image/png;base64,{img_b64}"}
                    ]
                }
            ],
            "max_tokens": 1000,
        }

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(GROQ_API_URL, headers=headers, json=payload) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data["choices"][0]["message"]["content"].strip()
                else:
                    print(f"[Groq Vision] Bad status {resp.status}")
                    print(await resp.text())
                    return None
    except Exception as e:
        print(f"[Groq Vision Error] {e}")
        return None


def _extract_json(text: str) -> Optional[Dict]:
    raw = (text or "").strip()
    if not raw:
        return None

    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        pass

    match = re.search(r"\{[\s\S]*\}", raw)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        return None


async def classify_image_with_groq(image_bytes: bytes) -> Dict:
    result = await read_image_with_groq(
        image_bytes,
        question=(
            "Classify the image into one type only: floor_plan, room_photo, document, other. "
            "Return strict JSON only with keys: type, confidence, reason."
        ),
    )
    parsed = _extract_json(result or "")
    if not parsed:
        return {"type": "other", "confidence": 0.0, "reason": "classification_parse_failed"}

    image_type = str(parsed.get("type") or "other").strip().lower()
    if image_type not in {"floor_plan", "room_photo", "document", "other"}:
        image_type = "other"

    try:
        confidence = float(parsed.get("confidence", 0.0))
    except Exception:
        confidence = 0.0

    return {
        "type": image_type,
        "confidence": max(0.0, min(1.0, confidence)),
        "reason": str(parsed.get("reason") or "").strip(),
    }


async def extract_floorplan_graph_with_groq(image_bytes: bytes, *, building: str, floor: str) -> Dict:
    result = await read_image_with_groq(
        image_bytes,
        question=(
            "Extract a topological navigation graph from this architectural floor plan. "
            "Return strict JSON only with schema: "
            "{\"nodes\":[{\"id\":\"string\",\"name\":\"string\",\"building\":\"string\",\"floor\":\"string\","
            "\"kind\":\"room|lab|hallway|entrance|office|stairs|other\",\"has_open_pc\":true|false,\"aliases\":[\"...\"]}],"
            "\"edges\":[{\"from\":\"node_id\",\"to\":\"node_id\",\"door\":\"optional\",\"weight\":1.0}]}. "
            "If unsure, return best-effort graph. "
            f"Use building='{building}' and floor='{floor}' when missing."
        ),
    )
    parsed = _extract_json(result or "")
    if not parsed:
        return {"nodes": [], "edges": []}

    nodes = parsed.get("nodes", [])
    edges = parsed.get("edges", [])
    if not isinstance(nodes, list):
        nodes = []
    if not isinstance(edges, list):
        edges = []

    normalized_nodes = []
    for node in nodes:
        if not isinstance(node, dict):
            continue
        normalized_nodes.append(
            {
                "id": str(node.get("id") or "").strip(),
                "name": str(node.get("name") or "").strip(),
                "building": str(node.get("building") or building).strip() or building,
                "floor": str(node.get("floor") or floor).strip() or floor,
                "kind": str(node.get("kind") or "room").strip().lower(),
                "has_open_pc": bool(node.get("has_open_pc", False)),
                "aliases": node.get("aliases", []) if isinstance(node.get("aliases", []), list) else [],
                "x": node.get("x"),
                "y": node.get("y"),
            }
        )

    normalized_edges = []
    for edge in edges:
        if not isinstance(edge, dict):
            continue
        normalized_edges.append(
            {
                "from": str(edge.get("from") or "").strip(),
                "to": str(edge.get("to") or "").strip(),
                "door": str(edge.get("door") or "").strip(),
                "weight": edge.get("weight", edge.get("distance", 1.0)),
            }
        )

    return {"nodes": normalized_nodes, "edges": normalized_edges}


async def extract_room_status_with_groq(image_bytes: bytes, *, default_building: str, default_floor: str) -> Dict:
    result = await read_image_with_groq(
        image_bytes,
        question=(
            "Extract room status details from this CICT image/document. "
            "Return strict JSON only with schema: "
            "{\"room_name\":\"string\",\"building\":\"string\",\"floor\":\"string\","
            "\"is_lab\":true|false,\"has_open_pc\":true|false,\"confidence\":0.0}. "
            "If unknown, keep empty strings and confidence 0.0."
        ),
    )
    parsed = _extract_json(result or "")
    if not parsed:
        return {
            "room_name": "",
            "building": default_building,
            "floor": default_floor,
            "is_lab": False,
            "has_open_pc": False,
            "confidence": 0.0,
        }

    try:
        confidence = float(parsed.get("confidence", 0.0))
    except Exception:
        confidence = 0.0

    return {
        "room_name": str(parsed.get("room_name") or "").strip(),
        "building": str(parsed.get("building") or default_building).strip() or default_building,
        "floor": str(parsed.get("floor") or default_floor).strip() or default_floor,
        "is_lab": bool(parsed.get("is_lab", False)),
        "has_open_pc": bool(parsed.get("has_open_pc", False)),
        "confidence": max(0.0, min(1.0, confidence)),
    }



