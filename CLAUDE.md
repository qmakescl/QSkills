# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

QSkills is an AI Agent Skills Collection — modular, self-contained skill packages that AI agents (Claude, OpenAI Code Interpreter, LangChain, etc.) use to perform statistical data analysis. The project language is Korean. Python 3.12+, managed with `uv`.

## Commands

```bash
# Install dependencies
uv sync

# Run df-basic-stats skill
python skills/df-basic-stats/scripts/compute_stats.py <input_file>
python skills/df-basic-stats/scripts/compute_stats.py <input_file> --no-html --no-md

# Run mean-comparison-test skill
python skills/mean-comparison-test/scripts/run_analysis.py \
  --data <file> --dv <dependent_var> --iv <independent_var> \
  --iv2 <post_var_wide>   --id_col <id_var_long> \
  --test_type independent|paired|anova \
  --alternative two-sided|greater|less \
  --equal_var true|false --output_dir ./results
```

There is no formal test runner. Evaluation specs live in `skills/*/evals/evals.json` as prompt+assertion pairs for manual or agent-driven validation.

## Architecture

Each skill is a self-contained directory under `skills/`:

```
skills/<skill-name>/
├── SKILL.md          # Interface contract: triggers, workflow, input/output schema
├── scripts/          # Executable Python scripts (CLI + importable module)
├── references/       # Algorithm docs, output schemas, report templates
├── evals/            # Evaluation cases (evals.json) and test data files
└── assets/           # Output artifacts (charts, reports)
```

**SKILL.md** is the primary specification document for each skill — it defines when the skill triggers, the step-by-step workflow, input/output formats, and edge case handling. Always read SKILL.md first when working on a skill.

### Key Design Patterns

- **Graceful degradation**: Core analysis runs on pandas/numpy only. Optional libraries (ydata-profiling, matplotlib, scipy, statsmodels, scikit-posthocs) are imported with try/except — if missing, those features are silently skipped.
- **Platform-agnostic paths**: Skills use `SKILL_DIR`, `UPLOAD_DIR`, `OUTPUT_DIR` environment variables. No hard-coded paths. Falls back to CWD when env vars are absent.
- **Dual interface**: Scripts work both as CLI tools (`python script.py <args>`) and importable Python modules.
- **JSON-first output**: Structured JSON is the primary output; Markdown reports and HTML profiling are secondary formats.
- **80% threshold rule**: Type inference for object columns uses an 80% success rate threshold for numeric/datetime conversion attempts.

### Current Skills

| Skill | Script | Purpose |
|-------|--------|---------|
| `df-basic-stats` | `compute_stats.py` | Auto type inference → per-type descriptive statistics → ydata-profiling HTML |
| `mean-comparison-test` | `run_analysis.py` | t-tests (independent/paired/Welch's), ANOVA, post-hoc (Tukey/Scheffé/Duncan/Dunnett's T3), effect sizes, distribution charts |

## Dependencies

Core: `pandas>=3.0.0`, `numpy>=2.4.0`

Optional (used by skills at runtime, not in pyproject.toml):
- `scipy`, `statsmodels`, `pingouin`, `scikit-posthocs` — statistical testing
- `matplotlib`, `seaborn` — visualization
- `ydata-profiling` — interactive HTML profiling reports
