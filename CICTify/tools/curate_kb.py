from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cictify_core.config import pdf_paths
from cictify_core.corpus_curation import curate_corpus


if __name__ == "__main__":
    payload = curate_corpus(pdf_paths())
    print(json.dumps(payload["report"], indent=2, ensure_ascii=True))
    print(f"Saved curated corpus to: {Path('vectorstore/curated_corpus.json').resolve()}")
    print(f"Saved report to: {Path('evaluation/kb_curation_report.md').resolve()}")
