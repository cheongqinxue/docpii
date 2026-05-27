# AGENTS.md — docpii agent briefing

Read this file at the start of any task in this repository. It is the durable project context
across chat sessions.

## Mission

Build a systematic pipeline to **discover PII / PCI / PHI in documents** using
[NuExtract3](https://huggingface.co/numind/NuExtract3) (structured JSON extraction from
images/text), with evaluation harnesses and an optional **QLoRA fine-tuning** path.

Primary output schema: entity-centric JSON (`DocumentExtraction` in `src/docpii/schemas/pii.py`)
where each person/org carries nested sensitive fields (IDs, contact, address, financial, health).

## Current phase

1. **Baseline eval** — measure NuExtract3 zero-shot on labeled documents.
2. **Data collection** — US-centric official ID samples from `.gov` sources (driver licenses,
   passports, USCIS cards, Medicare, etc.); synthetic rendering where no official image exists.
3. **Fine-tune later** — QLoRA once we have gold JSONL manifests.

## Repository map

| Path | Purpose |
|------|---------|
| `src/docpii/` | Installable Python package — reusable library code, CLI, tests |
| `scripts/` | Ad-hoc runnable scripts (see index below); **not** the main API surface |
| `configs/` | YAML defaults (vLLM, inference, training) |
| `data/raw/` | Source documents (gitignored) |
| `data/gold/` | Annotated ground truth (gitignored) |
| `data/samples/` | Small public examples |
| `tests/` | pytest suite |
| `outputs/` | Run artifacts (gitignored) |
| `uv.lock` | Locked dependencies (commit this file) |
| `.python-version` | Python pin for uv (3.11) |

### Package vs scripts

- Put **durable, tested, reusable** logic in `src/docpii/`.
- Put **one-off or operational** helpers in `scripts/` (downloaders, batch converters, figure
  extractors from PDFs, etc.).
- Before adding a script, **check the script index below** and extend an existing script if it
  already covers the need.

## Agent workflow

1. Read this file (`AGENTS.md`).
2. Read `README.md` for install/run commands if needed.
3. **Review the script index** before creating any new file under `scripts/`.
4. Prefer extending `src/docpii/` or an existing script over duplicating logic.
5. After adding, removing, or materially changing a script, **update the script index** in
   this file in the same change.
6. Keep the script library at **≤ 20 scripts** (see `.cursor/rules/scripts-library-limit.mdc`).

## Data policy

- Use **official public samples** (`.gov` educational/reference materials) or **clearly
  synthetic** documents we generate.
- Do **not** scrape real individuals' identity documents from the web.
- Do **not** produce counterfeit-grade document forgeries.
- Ground truth uses placeholder values visible on official samples (e.g. `JOHN DOE`).

## Key commands

Uses [uv](https://docs.astral.sh/uv/) — see `.cursor/rules/uv-tooling.mdc`.

```bash
# Install / sync env from uv.lock
uv sync

# Serve NuExtract3 (GPU + vLLM extra required)
uv sync --extra vllm
bash scripts/serve_nuextract.sh

# CLI
uv run docpii template
uv run docpii extract path/to/doc.pdf -o outputs/out.json
uv run docpii eval data/samples/manifest.example.jsonl -r outputs/eval_report.json

# Dev
uv run pytest
uv run ruff check src tests scripts
uv run ruff format src tests scripts

# Official US ID reference samples (.gov catalog)
uv run python scripts/fetch_official_samples.py

# After pyproject.toml dependency changes
uv lock && uv sync
```

## Extraction schema (summary)

Top-level: `DocumentExtraction` → `entities[]` → nested `contact`, `address`,
`government_ids`, `financial_accounts`, `health_identifiers`.

NuExtract templates are derived from Pydantic via `src/docpii/templates/nuextract.py`.

Eval manifests: JSONL with `id`, `document`, `ground_truth` (see `data/samples/manifest.example.jsonl`).

---

## Script library index

**Maintained inventory of `scripts/`.** Count: **2 / 20**.

Review this table before writing a new script. If a script is removed or merged, update this
section in the same commit.

| Script | Language | Purpose | When to use |
|--------|----------|---------|-------------|
| `serve_nuextract.sh` | bash | Launch vLLM serving `numind/NuExtract3` on port 8000 | Local inference API for `docpii extract` / eval |
| `fetch_official_samples.py` | python | Download US `.gov` ID reference images/PDFs from catalog | Seed `data/raw/official_samples/` for baseline eval |

### Index changelog

| Date | Change |
|------|--------|
| 2026-05-27 | Initial index with `serve_nuextract.sh` |
| 2026-05-27 | Added `fetch_official_samples.py` and `data/sources/us_official_id_catalog.yaml` |

### Planned scripts (not yet implemented)

| Planned | Purpose |
|---------|---------|
| `build_manifest_from_labels.py` | Convert span/field labels → `DocumentExtraction` JSONL |

---

## Coding standards

See `.cursor/rules/python-standards.mdc`. Summary:

- DRY — reuse `src/docpii/` and existing scripts.
- Google-style docstrings on public functions/classes.
- Type hints, ruff (line length 100), pytest for tests.
- Minimize cross-module coupling; scripts should be thin wrappers over `docpii` where possible.

## References

- [NuExtract3 model card](https://huggingface.co/numind/NuExtract3)
- [NuExtract fine-tuning cookbooks](https://github.com/numindai/nuextract/tree/main/cookbooks)
