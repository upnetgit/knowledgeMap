# Copyright 2021 JD.com, Inc., JD AI
"""
@author: Yehao Li
@contact: yehaoli.sysu@gmail.com
"""
from .build import build_evaluation

from .coco_evaler import COCOEvaler
from .coco_printer import COCOPrinter
from .vqa_evaler import VQAEvaler
from .vcr_evaler import VCREvaler
from .retrieval_evaler import RetrievalEvaler

try:
	from .relation_eval import compute_metrics, load_jsonl, save_report
except Exception:  # pragma: no cover - optional helper should not break base evaluation imports
	compute_metrics = None
	load_jsonl = None
	save_report = None

__all__ = [
	"build_evaluation",
	"COCOEvaler",
	"COCOPrinter",
	"VQAEvaler",
	"VCREvaler",
	"RetrievalEvaler",
	"compute_metrics",
	"load_jsonl",
	"save_report",
]

