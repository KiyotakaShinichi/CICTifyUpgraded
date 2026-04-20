from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cictify_core.spatial_graph import SpatialGraphStore


def _load_cases(path: Path) -> List[Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        return []
    return [item for item in payload if isinstance(item, dict)]


def _case_result(store: SpatialGraphStore, case: Dict[str, Any]) -> Dict[str, Any]:
    query = str(case.get("query") or "").strip()
    expected = [str(t).strip().lower() for t in case.get("expect_contains", []) if str(t).strip()]

    answer = store.answer_navigation_query(query)
    answer_low = (answer or "").lower()
    matched = [token for token in expected if token in answer_low]
    missing = [token for token in expected if token not in answer_low]

    passed = bool(answer) and not missing
    return {
        "id": str(case.get("id") or ""),
        "query": query,
        "expected": expected,
        "answer": answer,
        "matched": matched,
        "missing": missing,
        "passed": passed,
    }


def run_evaluation(cases_path: Path, output_path: Path) -> Dict[str, Any]:
    cases = _load_cases(cases_path)
    store = SpatialGraphStore()

    results = [_case_result(store, case) for case in cases]
    total = len(results)
    passed = sum(1 for r in results if r.get("passed"))
    failed = total - passed
    accuracy = round((passed / total) * 100.0, 2) if total else 0.0

    payload = {
        "timestamp": datetime.now().isoformat(),
        "cases_path": str(cases_path),
        "spatial_graph_stats": store.stats(),
        "summary": {
            "total": total,
            "passed": passed,
            "failed": failed,
            "accuracy_percent": accuracy,
        },
        "results": results,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate spatial query behavior against fixed cases.")
    parser.add_argument(
        "--cases",
        default=str(ROOT / "evaluation" / "spatial_eval_cases.json"),
        help="Path to evaluation cases JSON file.",
    )
    parser.add_argument(
        "--output",
        default=str(ROOT / "evaluation" / "spatial_eval_results.json"),
        help="Path for evaluation results JSON output.",
    )
    args = parser.parse_args()

    payload = run_evaluation(Path(args.cases), Path(args.output))
    summary = payload.get("summary", {})
    print("Spatial Evaluation")
    print(f"Total: {summary.get('total', 0)}")
    print(f"Passed: {summary.get('passed', 0)}")
    print(f"Failed: {summary.get('failed', 0)}")
    print(f"Accuracy: {summary.get('accuracy_percent', 0.0)}%")
    print(f"Saved: {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
