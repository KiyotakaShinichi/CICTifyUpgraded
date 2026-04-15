# KB Proofread Report (2026-04-15)

## Corpus Checked
- BulSU Student handbook.pdf
- BulSU-Enhanced-Guidelines.pdf
- CICT Rooms - Pics & Desc.docx.pdf
- CICT-Rooms.pdf
- CICTify - FAQs.pdf
- Faculty Manual for BOR.pdf
- guide.pdf
- UnivCalendar_2526.pdf
- UniversityCalendar_AY2526.pdf (empty)

## Automated Quality Signals (first ~20 pages sampled per file)
- BulSU Student handbook.pdf: single_char_ratio=0.060, generally readable.
- BulSU-Enhanced-Guidelines.pdf: single_char_ratio=0.099, moderate OCR spacing noise.
- CICT Rooms - Pics & Desc.docx.pdf: single_char_ratio=0.042, useful location facts but many short line fragments.
- CICT-Rooms.pdf: single_char_ratio=0.148, high noise and very short content; likely low-value duplicate of room/floorplan info.
- CICTify - FAQs.pdf: single_char_ratio=0.031, high-value factual snippets but text has broken separators.
- Faculty Manual for BOR.pdf: single_char_ratio=0.086, broad policy text, weak for person-name queries.
- guide.pdf: single_char_ratio=0.099, appears duplicated with BulSU-Enhanced-Guidelines.pdf.
- UnivCalendar_2526.pdf: single_char_ratio=0.093, moderate OCR spacing noise.
- UniversityCalendar_AY2526.pdf: empty file.

## Data Issues Driving Bad Answers
1. Duplicate policy source:
- guide.pdf duplicates BulSU-Enhanced-Guidelines.pdf, causing repeated retrieval chunks.

2. Empty file in corpus:
- UniversityCalendar_AY2526.pdf has 0 bytes.

3. Mixed-intent retrieval:
- Generic term overlap can pull policy/admin chunks for dean-name questions.

4. Fragmented OCR chunks:
- Room/location PDFs contain fragmented bullet lines; extraction can return partial phrases if not post-processed.

## Retrieval Improvements Applied
1. Source curation in corpus loading:
- If BulSU-Enhanced-Guidelines.pdf exists, guide.pdf is excluded.
- Empty PDFs are excluded automatically.

2. Manifest-driven FAISS rebuild:
- Index now rebuilds when source list/chunk settings change.

3. Query-aware source boosting:
- Dean/director/coordinator queries boost CICTify - FAQs.
- Transferee/shiftee/returnee/admission queries boost Enhanced Guidelines.
- Room/floor/lab/location queries boost CICT Rooms sources.

4. Early dedupe of retrieved chunks:
- Near-duplicate chunk content is removed before reranking.

5. Better fallback hierarchy:
- Direct fact fallback for dean name, transferee definition, and Acad 1 location.
- Extractive fallback remains as backup when generation is unavailable.

## Recommended Hyperparameters (Current Corpus)
- CHUNK_SIZE: 900
- CHUNK_OVERLAP: 120
- RETRIEVAL_K: 12
- RAG_RERANK_TOP_N: 5
- RAG_MIN_TERM_OVERLAP: 2

Reasoning:
- Slightly smaller chunks reduce mixed-topic contamination from long policy pages.
- Lower overlap reduces repeated near-identical passages.
- Slightly higher initial K helps recall across noisy OCR docs.
- Tighter rerank + minimum term overlap reduces off-topic chunks.

## Optional Next Data Cleanup (Highest ROI)
1. Remove CICT-Rooms.pdf if CICT Rooms - Pics & Desc.docx.pdf is already maintained.
2. Keep only one of BulSU-Enhanced-Guidelines.pdf and guide.pdf at file-system level.
3. Replace broken OCR PDFs with cleaner exports where possible.
4. Add a compact curated FAQ JSON for high-confidence facts (dean name, offices, room aliases) to bypass noisy PDF extraction.
