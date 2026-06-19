# Design Decision Log

All significant technical and methodological choices made during the
dissertation are recorded here with justification and alternatives considered.

Format for each entry:

```
## DEC-NNN — Short title
**Date:** YYYY-MM-DD
**Status:** Proposed | Accepted | Superseded

### Context
What problem or question prompted this decision?

### Decision
What was decided?

### Rationale
Why was this option chosen over the alternatives?

### Alternatives considered
- Alternative A — why rejected
- Alternative B — why rejected

### Consequences
What changes as a result? What becomes easier or harder?
```

---

## DEC-001 — Use Parquet for processed data artefacts
**Date:** 2026-06-19
**Status:** Accepted

### Context
The raw Excel file is slow to parse on every pipeline run. Intermediate
dataframes need to be persisted between stages.

### Decision
Write all processed artefacts as Apache Parquet files via `pandas.to_parquet`.

### Rationale
Parquet is columnar, compressed, and preserves dtypes (including `datetime64`
and `Categorical`). It is an order of magnitude faster to read than CSV and
does not lose numeric precision unlike CSV.

### Alternatives considered
- CSV — loses dtypes, slower I/O, larger files
- Pickle — not human-inspectable, Python-version-dependent

### Consequences
Requires `pyarrow` or `fastparquet` (pulled in transitively by `pandas`).
Processed files are excluded from git via `.gitignore`.

---

## DEC-002 — Centralise all configuration in `src/config.py`
**Date:** 2026-06-19
**Status:** Accepted

### Context
Hyper-parameters, file paths, and constants were risk of being scattered
across multiple modules, making reproducibility and tuning difficult.

### Decision
All tunable values live exclusively in `src/config.py` and are imported
from there; no module defines its own magic numbers.

### Rationale
Single source of truth; a reviewer or examiner can inspect one file to
understand all choices. Facilitates grid-search or sensitivity analysis
by changing one file.

### Alternatives considered
- YAML / TOML config file — adds a parsing dependency and separates config
  from the type-annotated Python context where it is used.
- Environment variables — appropriate for secrets, not for algorithm constants.

### Consequences
Every pipeline module has a dependency on `src/config.py`.

---

<!-- Add new decisions below this line -->
