# GitHub Release Notes

This repository version is intended for source-code publication and reproducible
simulation runs.

Included:

- Source modules under `src/`
- YAML configs under `configs/`
- Topic-based command-line scripts under `scripts/`
- Minimal notebooks under `notebooks/`
- Tests under `tests/`
- Topic-renamed legacy source files under `legacy_code/PYTHON/`

Excluded from the GitHub copy:

- Generated `results/` data, logs, and figures
- Legacy generated `.npz`, `.png`, and parameter JSON outputs
- Local caches and bytecode
- Raw experimental data

Before public release, confirm:

- Whether the project should use an open-source license.
- Whether thesis-specific legacy code should remain included.
- Whether any unpublished experimental data or private paths appear in outputs.
