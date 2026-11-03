# Patched SQLite runtime build

The rc.3 Companion uses CPython 3.13.18 / PBS 20261026 with **SQLite 3.54.1**.
The original upstream distribution contains SQLite 3.53.1. This is a project-built
derivative, recorded explicitly in `runtime/SQLITE_BUILD.json`.

`release/sqlite-runtime.json` pins the official SQLite source by size, SHA-256 and
the owning release's SHA3-256. SQLite's source is public domain; the existing
`LICENSE.sqlite.txt` notice is retained. The source ID is checked against the
actual loaded engine. See [the release](https://sqlite.org/releaselog/3_54_1.html)
and [SQLite's licensing statement](https://sqlite.org/copyright.html).

On macOS and Linux the build uses the exact PBS full archive's object inventory,
the matching pinned Astral LLVM compiler and rebuilt SQLite. It relinks both the
Python executable and shared libpython, because replacing only the latter leaves
the original executable's embedded engine unchanged. The macOS deployment target
and stack size remain explicit, as do Python C API exports, dylib versions and
the executable-relative library path. On Windows the existing MSVC x64 toolchain builds
the DLL loaded by `_sqlite3.pyd`, preserving PBS's feature definitions.

Every build runs native FTS3/4/5, rtree, geopoly, dbstat, serialization and shared
library checks, native extension imports, then CPython's `test_sqlite3` suite. CPython development test
packages and unused libpython linking archives are removed afterward. The derivative receipt binds source, recipe,
compiler inputs, native payload hashes and test results. Companion packaging
refuses a missing/stale receipt, modified native payload or wrong loaded engine.

Original `release/runtime-legal/*/PYTHON.json` files and notices describe the PBS
base and retain their upstream bytes. The derivative receipt describes the
replacement; it does not rewrite upstream metadata. Python, OpenSSL and other
component licenses remain separate from the project's MIT license. Compilers and
build trees are not shipped in player bundles.

Build from an authorized clean source checkout with the platform's native build
tools available:

```sh
uv run --frozen python scripts/fetch_companion_runtime.py --output artifacts/patched-python
```

macOS requires Xcode command-line tools; Windows requires an MSVC x64 developer
environment; Linux requires native build tools and zstd-capable tar. These are
release-builder requirements. Players receive the completed runtime.
