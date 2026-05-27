# docpii

Systematic **PII / PCI / PHI discovery** from document images using [NuExtract3](https://huggingface.co/numind/NuExtract3), with evaluation harnesses and a QLoRA fine-tuning path.

NuExtract3 accepts a JSON template (schema) and returns structured JSON extracted from text or images — a good fit for entity-centric sensitive-data discovery before remediation.

**Agents:** see [`AGENTS.md`](AGENTS.md) for durable project context, data policy, and the `scripts/` library index.

## Project layout

```
vlm/
├── configs/                 # YAML defaults (vLLM, inference, training)
├── data/
│   ├── raw/                 # Unlabeled source documents (gitignored)
│   ├── gold/                # Annotated ground truth (gitignored)
│   └── samples/             # Example manifest + sample document
├── AGENTS.md                # Agent briefing + scripts index (read first)
├── scripts/                 # Ad-hoc scripts (indexed in AGENTS.md, max 20)
│   └── serve_nuextract.sh   # Launch vLLM with NuExtract3
├── src/docpii/
│   ├── schemas/             # Pydantic models (Entity, DocumentExtraction, …)
│   ├── templates/           # Pydantic → NuExtract JSON template
│   ├── inference/           # vLLM client + document loading
│   ├── evaluation/          # Benchmark runner + NuExtract-style metrics
│   ├── data/                # Training/eval dataset formats
│   ├── training/            # QLoRA fine-tuning entry point
│   └── cli.py               # `docpii` command-line tool
└── tests/
```

## Quick start

### 1. Install

Requires [uv](https://docs.astral.sh/uv/getting-started/installation/).

```bash
cd /home/cheong/Desktop/Work/vlm
uv sync
cp .env.example .env
```

Optional extras:

```bash
uv sync --extra pdf          # PDF page rendering (PyMuPDF)
uv sync --extra vllm         # Local NuExtract3 server (GPU + CUDA; large install)
uv sync --extra train        # QLoRA fine-tuning stack
uv sync --all-extras         # everything above (very large; may need GPU-matched torch/vllm)
```

`vllm` requires a **CUDA-capable GPU** and compatible driver. Install it only on the machine
that serves the model, then start the API:

```bash
uv sync --extra vllm
bash scripts/serve_nuextract.sh
# Optional: ENABLE_MTP=true bash scripts/serve_nuextract.sh
```

Run commands through uv (no manual `source .venv/bin/activate` needed):

```bash
uv run docpii template
uv run pytest
```

### Official US ID reference samples

Download curated `.gov` educational samples (driver licenses, passports, USCIS cards, Medicare):

```bash
uv run python scripts/fetch_official_samples.py
```

Catalog: `data/sources/us_official_id_catalog.yaml`  
Provenance index: `data/samples/official_download_index.json`  
Files: `data/raw/official_samples/` (gitignored)

### 2. Serve NuExtract3 (vLLM)

Requires a GPU, CUDA, and the `vllm` extra:

```bash
uv sync --extra vllm
bash scripts/serve_nuextract.sh
# API at http://localhost:8000/v1
```

### 3. Inspect the extraction schema

The default schema captures **entities** (person, org, …) and nested sensitive fields (government IDs, financial accounts, health identifiers, contact, address):

```bash
uv run docpii template
```

### 4. Run extraction on a document

```bash
uv run docpii extract path/to/document.pdf -o outputs/extraction.json
```

### 5. Systematic evaluation

Create a manifest (see `data/samples/manifest.example.jsonl`). Each row links a document to ground-truth JSON:

```bash
uv run docpii eval data/samples/manifest.example.jsonl -r outputs/eval_report.json
```

The evaluator uses tree-aligned leaf scoring similar to NuExtract's internal benchmark (indel similarity for `verbatim-string` fields, exact match otherwise).

### 6. Fine-tune with QLoRA (optional)

Prepare labeled JSONL (`id`, `document`, `ground_truth`), then:

```bash
uv run docpii train data/gold/train.jsonl --output-dir checkpoints/docpii-qlora
```

The training module is a scaffold around TRL `SFTTrainer` — you'll likely want to customize the multimodal collator as your dataset grows.

## Extraction schema

`DocumentExtraction` is the top-level output:

| Field | Description |
|-------|-------------|
| `document_id` | External doc ID if present |
| `entities[]` | Detected entities with nested PII |

Each `Entity` includes:

- **Identity**: `entity_type`, `name`, `role_or_context`
- **Contact**: email, phone, fax, website
- **Address**: structured or `full_address`
- **Government IDs**: SSN, passport, driver's license, …
- **Financial**: credit/debit cards, bank accounts, IBAN, …
- **Health (PHI)**: MRN, insurance IDs, NPI, …

Extend `src/docpii/schemas/pii.py` as your remediation taxonomy evolves.

## Workflow for your use case

1. **Baseline** — Run `docpii eval` on a small labeled set to measure NuExtract3 zero-shot on your document types.
2. **Iterate schema** — Tune Pydantic models and `instructions` until the template matches how you want entities grouped.
3. **Grow gold data** — Annotate real (redacted/synthetic) documents into `data/gold/`.
4. **Fine-tune** — QLoRA on NuExtract3 with your JSONL; re-evaluate with the same manifest format.
5. **Deploy** — Merge adapters or serve base + LoRA via vLLM for production discovery pipelines.

## Development

```bash
uv run ruff check src tests scripts
uv run ruff format src tests scripts
uv run mypy src
uv run pytest
```

## References

- [NuExtract3 model card](https://huggingface.co/numind/NuExtract3)
- [NuExtract fine-tuning cookbooks](https://github.com/numindai/nuextract/tree/main/cookbooks)
- [numind Python SDK](https://pypi.org/project/numind/) — `convert_json_schema_to_nuextract_template`
