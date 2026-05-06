from .tokenization_bert import BertTokenizer

try:
	from .teaching_tokenizer import TeachingTokenizer, detect_stage_tags, normalize_teaching_terms
except Exception:  # pragma: no cover - optional helper should not break base tokenizer imports
	TeachingTokenizer = None

	def detect_stage_tags(text):
		return []

	def normalize_teaching_terms(text):
		return str(text or "")

__all__ = ["BertTokenizer", "TeachingTokenizer", "detect_stage_tags", "normalize_teaching_terms"]
