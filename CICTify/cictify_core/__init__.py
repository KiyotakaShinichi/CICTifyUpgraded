from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from .floorplan_context import FloorplanContextStore
	from .ocr_ingestion import OCRIngestion
	from .orchestrator import CascadeOrchestrator, CascadeResult
	from .spatial_graph import SpatialGraphStore
	from .spatial_tool import SpatialAwarenessTool

__all__ = [
	"CascadeOrchestrator",
	"CascadeResult",
	"OCRIngestion",
	"FloorplanContextStore",
	"SpatialGraphStore",
	"SpatialAwarenessTool",
]


def __getattr__(name: str):
	if name in {"CascadeOrchestrator", "CascadeResult"}:
		from .orchestrator import CascadeOrchestrator, CascadeResult
		return {"CascadeOrchestrator": CascadeOrchestrator, "CascadeResult": CascadeResult}[name]
	if name == "OCRIngestion":
		from .ocr_ingestion import OCRIngestion
		return OCRIngestion
	if name == "FloorplanContextStore":
		from .floorplan_context import FloorplanContextStore
		return FloorplanContextStore
	if name == "SpatialGraphStore":
		from .spatial_graph import SpatialGraphStore
		return SpatialGraphStore
	if name == "SpatialAwarenessTool":
		from .spatial_tool import SpatialAwarenessTool
		return SpatialAwarenessTool
	raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
