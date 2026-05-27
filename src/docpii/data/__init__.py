"""Dataset formats for evaluation and QLoRA fine-tuning."""

from docpii.data.formats import (
    TrainingExample,
    build_training_messages,
    load_training_jsonl,
    write_training_jsonl,
)
from docpii.data.official_samples import (
    DEFAULT_CATALOG,
    DEFAULT_INDEX_PATH,
    DEFAULT_OUTPUT_DIR,
    DownloadReport,
    OfficialSampleSpec,
    download_official_samples,
    load_catalog,
)

__all__ = [
    "DEFAULT_CATALOG",
    "DEFAULT_INDEX_PATH",
    "DEFAULT_OUTPUT_DIR",
    "DownloadReport",
    "OfficialSampleSpec",
    "TrainingExample",
    "build_training_messages",
    "download_official_samples",
    "load_catalog",
    "load_training_jsonl",
    "write_training_jsonl",
]
