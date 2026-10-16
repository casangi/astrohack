# AGENT.md — AstroHACK

Guidance for AI agents (and humans) working on **AstroHACK** (Holography and
Antenna Commissioning Kit). Read this before making changes; it captures the
architecture, where the external-data (casacore) boundary lives, how to run the
tests, and the non-obvious gotchas.

---

## 1. What this package is

AstroHACK is a pure-Python package for **NRAO telescope support**: radio
holography (VLA / ALMA, with basic near-field ALMA and emerging ngVLA support)
and **antenna-position corrections** (locit). It ports the AIPS holography tasks
(UVHOL / HOLOG / PANEL) onto a modern **Dask + Numba** stack so reductions
parallelize and run fast.

Inputs are **MeasurementSet v2** (`.ms`) and CASA **calibration tables**;
outputs are self-describing **`.zarr`** datatrees wrapped by the package's own
"mds" file classes (see §2). Coordinate math uses **astropy**; gridding/fitting
use numpy/scipy/scikit-image/numba.

---

## 2. Architecture (three layers + helpers)

```
src/astrohack/
├── __init__.py                # exports the user-facing pipeline functions
├── <task>.py                  # USER-FACING api: param validation + dask graph build
│     extract_pointing, extract_holog, holog, panel, locit, extract_locit,
│     combine, beamcut, cassegrain_ray_tracing, image_comparison_tool
├── core/                      # COMPUTE: the per-chunk work the dask graph runs
│     extract_pointing.py, extract_holog.py, extract_locit.py,  ← casacore lives here
│     holog.py, panel.py, combine.py, beamcut.py, ...
├── io/                        # the "*_mds" file classes (zarr datatree wrappers)
│     dio.py, base_mds.py, holog_mds.py, point_mds.py, image_mds.py,
│     locit_mds.py, panel_mds.py, position_mds.py, beamcut_mds.py
├── antenna/                   # telescope + panel geometry/fitting classes
├── utils/                     # algorithms, gridding, conversion, fits, constants, ...
├── visualization/             # plot tools, dashboards
└── config/                    # parameter-json scaffolding
```

A typical pipeline call (e.g. `extract_holog`) lives in the top-level module,
validates parameters, then maps a `core/`-module chunk function over a Dask
graph (via `toolviper`/`graphviper`). The `core/` functions are where the MS is
actually read.

**The usual holography flow:** `extract_pointing` → `extract_holog` → `holog`
(grid + FFT to aperture) → `panel` (panel fitting). The position flow is
`extract_locit` → `locit`.

---

## 3. The casacore boundary — now **casacoretables**

**MS / cal-table reading is the ONLY use of casacore in the package, and it is
confined to exactly three files:**

- `src/astrohack/core/extract_holog.py`
- `src/astrohack/core/extract_pointing.py`
- `src/astrohack/core/extract_locit.py`

Each does `from casacoretables import tables as ctables` and uses only:
- `ctables.table(path, readonly=True, lockoptions={"option": "usernoread"}, ack=False)`
- `ctables.taql("select ... from $table_obj where ...")` (`$var` substitution)
- `.getcol(col[, startrow=, nrow=])`, `.close()`
- the `path::SUBTABLE` and `path/SUBTABLE` ways of opening MS subtables.

### Why casacoretables (not python-casacore / casatools)

`casacoretables` (`/Users/jsteeb/Dropbox/viper_dev/casacoretables`) is a
standalone, minimal build of casacore's **tables** layer with a pybind11 API
that mirrors `python-casacore`'s `casacore.tables`. It pulls in **no Boost** and
**no other casacore modules**, builds cleanly on **Linux and macOS**, and hides
its symbols so it can coexist in-process with CASA's `casatools`. This removes
AstroHACK's old macOS pain point (python-casacore had to be conda-installed
separately and was excluded from the wheel on darwin).

The migration was a **drop-in import swap** — the table API AstroHACK uses
(`table`, `taql`, `getcol`, `::` subtables, dict `lockoptions`) is identical, so
no call sites changed beyond the import line. casacoretables ships **only**
tables, but AstroHACK never needed `images`/`ms`/`measures`/`quanta`, so nothing
is missing. Coordinate conversions that *would* have used casacore measures are
done with **astropy** (already a core dependency; see `utils/conversion.py`,
`utils/fits.py`).

`utils/constants.py` has a casacore **URL in a comment** only — not an import.

### CASA-side example scripts (left untouched on purpose)

`etc/locit/casa/pre-locit-script.py` and `beamcut_CASA_calibration.py` use
`import casatools`; `etc/locit/casa/post-locit-plot-phases.py` uses
`from casacore import tables`. These are **standalone helper scripts meant to be
run inside a CASA session** (where casatools / python-casacore exist), not part
of the installed package or the test suite. Leave them on casatools/casacore.

---

## 4. Build / install / environment

**Use the conda `zinc` environment** (`conda activate zinc`) — it has
casacoretables and the scientific stack installed.

```bash
conda activate zinc
cd astrohack
pip install -e .          # setuptools src-layout; deps incl. casacoretables, toolviper, dask, numba, astropy, zarr<3
```

Packaging notes / pre-existing quirks (not introduced by the casacore work):
- No `[build-system]` table in `pyproject.toml` → setuptools default backend.
- `python` 3.12–3.14 (`requires-python >= 3.12, < 3.15`, issue #388).
- `zarr<3.0.0` is pinned deliberately (zarr 3 migration: issue #300), and
  therefore `xarray<2026.9.0` (xarray 2026.9 dropped zarr-python 2 support).
  numpy is not pinned: numba sets its own supported numpy range.
- zarr 2 needs `numcodecs<0.16`, which has no cp314 wheels, so on Python 3.14
  pip builds numcodecs 0.15.1 from source (needs GCC <= 14 or
  `CFLAGS=-std=gnu17`; GCC 15 defaults to C23).

For quick iteration without installing, `PYTHONPATH=src python -m pytest …`
works (the package imports cleanly from `src/`).

---

## 5. Tests

```bash
conda activate zinc
cd astrohack
PYTHONPATH=src python -m pytest tests/unit -q          # unit suite
PYTHONPATH=src python -m pytest tests/stakeholder -q   # heavier, downloads bigger MSs
```

- Test data is **downloaded on demand** via `toolviper.utils.data.download(...)`
  in each test's `setup_class` (so the unit suite needs network the first time;
  it caches under a per-test `*_data/` folder and `teardown_class` removes it).
- The casacoretables boundary is exercised end-to-end by
  `tests/unit/user_facing_functions/test_extract_{pointing,holog,locit}.py` —
  run those first when changing anything in the three `core/extract_*.py` files.
- A pre-existing `NumbaPendingDeprecationWarning` (reflected list) is harmless.

CI (`.github/workflows/python-testing-{linux,macos}.yml`) calls the shared
`nrao/gh-actions-templates-public` reusable workflows, so the build/test steps
are defined **outside this repo**. With casacoretables now an ordinary
dependency, `pip install .` resolves it from PyPI on every supported
interpreter: `casacoretables>=0.0.3` ships cp311/cp312/cp313 wheels for
`manylinux_2_28 x86_64` and `macosx_11_0 arm64` (plus an sdist that needs
bison≥3/flex to build, e.g. on Intel macOS or Linux aarch64). The macOS
template's separate python-casacore install step is no longer needed.

---

## 6. Gotchas

- **Dropbox + casacore table locking.** This tree lives under Dropbox, whose
  sync interferes with casacore lock files ("table is in use" / "Directory not
  empty"). The code already opens tables with `lockoptions={"option":
  "usernoread"}` and `readonly=True`, which avoids most of it. If you write a
  scratch table in a test, put it in `$TMPDIR`, not the Dropbox tree.
- **`conda activate` in non-interactive shells** may not actually switch the
  `python` on `PATH`. Verify with `which python` / use the absolute interpreter
  `/Users/jsteeb/miniforge3/envs/zinc/bin/python` when in doubt.
- **String vs numeric columns.** `getcol` on a string MS column returns a Python
  `list` (so `.index()`/`.pop()` are used on names/stations), while numeric
  columns return numpy arrays (`.tolist()` is called explicitly). This matches
  python-casacore and is what casacoretables reproduces — don't "normalize" one
  into the other.
- **Numba `@njit(cache=...)` functions** (`njit_caching` in
  `utils/constants.py`) hold the hot loops (e.g. `_extract_holog_chunk_jit`).
  Type changes to the arrays they receive can trigger recompiles or reflected-
  list warnings.

---

## 7. Quick reference

```bash
conda activate zinc

# install (editable) and import-check the casacore boundary
pip install -e .
python -c "import astrohack.core.extract_holog, astrohack.core.extract_pointing, astrohack.core.extract_locit; print('ok')"

# the three tests that exercise casacoretables end-to-end
PYTHONPATH=src python -m pytest tests/unit/user_facing_functions/test_extract_pointing.py \
    tests/unit/user_facing_functions/test_extract_holog.py \
    tests/unit/user_facing_functions/test_extract_locit.py -q

# full unit suite
PYTHONPATH=src python -m pytest tests/unit -q
```

Sibling package using the same backend: **xradio**
(`/Users/jsteeb/Dropbox/viper_dev/xradio`) — also migrated to casacoretables.
Backend repo + its own AGENTS.md: `/Users/jsteeb/Dropbox/viper_dev/casacoretables`.
