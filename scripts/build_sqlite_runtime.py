"""Build-time SQLite replacement in a pinned PBS runtime, with native verification.

POSIX reconstructs both Python executables and shared libpython from upstream
objects. Windows rebuilds the DLL used by _sqlite3.pyd. No import hooks, pip
replacement module or system SQLite is used. Original PBS notices stay intact.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from scripts.companion_downloads import download, extract, platform_key

ROOT = Path(__file__).resolve().parents[1]
POLICY = Path("release/sqlite-runtime.json")
RECEIPT = "SQLITE_BUILD.json"
DEFINES = (
    "SQLITE_ENABLE_DBSTAT_VTAB",
    "SQLITE_ENABLE_FTS3",
    "SQLITE_ENABLE_FTS3_PARENTHESIS",
    "SQLITE_ENABLE_FTS4",
    "SQLITE_ENABLE_FTS5",
    "SQLITE_ENABLE_GEOPOLY",
    "SQLITE_ENABLE_RTREE",
)
PROBE = """import ctypes, json, pathlib, sqlite3, sys
# Compiled Python extensions resolve these through the executable's C API.
# SQLite's builtin module alone cannot detect an accidentally stripped export table.
assert ctypes.pythonapi.PyCapsule_GetPointer
ctypes.c_void_p.in_dll(ctypes.pythonapi, 'PyBaseObject_Type')
c = sqlite3.connect(':memory:')
for engine in ('fts3', 'fts4', 'fts5'):
    c.execute('create virtual table t_' + engine + ' using ' + engine + '(text)')
    c.execute('insert into t_' + engine + ' values (?)', ('working search',))
    assert c.execute('select text from t_' + engine + ' where text match ?',
                     ('working',)).fetchone() == ('working search',)
c.execute('create virtual table bounds using rtree(id, lo, hi)')
c.execute('create virtual table polygons using geopoly')
c.execute('select * from dbstat').fetchall()
c.execute('create table ordinary(value blob)')
c.execute('insert into ordinary values (?)', (b'roundtrip',))
copy = sqlite3.connect(':memory:')
copy.deserialize(c.serialize())
assert copy.execute('select value from ordinary').fetchone() == (b'roundtrip',)
library = ('DLLs/sqlite3.dll' if sys.platform == 'win32' else
           'lib/libpython3.13.dylib' if sys.platform == 'darwin' else
           'lib/libpython3.13.so.1.0')
shared = ctypes.CDLL(str(pathlib.Path(sys.prefix) / library))
shared.sqlite3_libversion.restype = ctypes.c_char_p
assert shared.sqlite3_libversion().decode() == sqlite3.sqlite_version
print(json.dumps({'version': sqlite3.sqlite_version,
 'source_id': c.execute('select sqlite_source_id()').fetchone()[0],
 'compile_options': [r[0] for r in c.execute('pragma compile_options')],
 'checks': ['python_c_api', 'fts3', 'fts4', 'fts5', 'rtree', 'geopoly', 'dbstat',
            'serialize', 'shared_library']}))
"""


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _run(args: list[str | Path], *, cwd: Path, env: dict[str, str] | None = None) -> str:
    environment = dict(os.environ if env is None else env)
    for name in list(environment):
        if name.startswith(("PYTHON", "DYLD_", "LD_")):
            environment.pop(name)
    result = subprocess.run(
        [str(arg) for arg in args],
        cwd=cwd,
        env=environment,
        check=False,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=1200,
    )
    if result.returncode:
        raise ValueError(f"{Path(args[0]).name} failed: {result.stdout[-3000:]}")
    return result.stdout


def _python(prefix: Path) -> Path:
    return prefix / ("python.exe" if sys.platform == "win32" else "bin/python3.13")


def probe(prefix: Path) -> dict[str, Any]:
    result: dict[str, Any] = json.loads(
        _run([_python(prefix), "-I", "-S", "-c", PROBE], cwd=prefix)
    )
    return result


def _modified_files(key: str) -> tuple[str, ...]:
    if key == "windows-x86_64":
        return ("DLLs/sqlite3.dll",)
    return (
        "bin/python3.13",
        "lib/libpython3.13.dylib" if key == "macos-arm64" else "lib/libpython3.13.so.1.0",
    )


def verify_runtime(prefix: Path, source: Path = ROOT) -> dict[str, Any]:
    """Bind the derivative's recipe and native payload, then exercise stdlib SQLite."""
    policy = json.loads((source / POLICY).read_bytes())
    receipt = json.loads((prefix / RECEIPT).read_bytes())
    key = platform_key()
    if (
        receipt.get("schema") != "neurocraft.sqlite-build.v1"
        or receipt.get("platform") != key
        or receipt.get("policy_sha256") != sha256(source / POLICY)
        or receipt.get("recipe_sha256") != sha256(source / "scripts/build_sqlite_runtime.py")
        or receipt.get("sqlite_source_sha256") != policy["source"]["sha256"]
    ):
        raise ValueError("SQLite runtime recipe/provenance does not match this candidate")
    files = receipt.get("modified_files", {})
    if set(files) != set(_modified_files(key)):
        raise ValueError("SQLite runtime native payload inventory is incomplete")
    for name, expected in files.items():
        path = prefix / name
        if path.is_symlink() or not path.is_file() or sha256(path) != expected:
            raise ValueError("SQLite runtime native payload changed: " + name)
    actual = probe(prefix)
    if actual["version"] != policy["version"] or actual["source_id"] != policy["source_id"]:
        raise ValueError("SQLite runtime is not the exact patched engine")
    return {
        "version": actual["version"],
        "source_id": actual["source_id"],
        "build_receipt_sha256": sha256(prefix / RECEIPT),
        "checks": actual["checks"],
    }


def _archive(pin: dict[str, Any], cache: Path, suffix: str) -> Path:
    path: Path = cache / (pin["sha256"] + suffix)
    download(pin, path)
    return path


def _extract_full(archive: Path, destination: Path) -> None:
    # Only exact hash-verified upstream build archives reach native tar. They are
    # build inputs, never user-supplied archives or runtime setup requests.
    destination.mkdir()
    _run(["tar", "-xf", archive, "-C", destination], cwd=destination)
    for path in destination.rglob("*"):
        if path.is_symlink() and not path.resolve().is_relative_to(destination.resolve()):
            raise ValueError("Upstream build archive contains an external symlink")


def _posix_build(
    stage: Path, cache: Path, policy: dict[str, Any], key: str
) -> tuple[Path, dict[str, Any]]:
    legal = json.loads((ROOT / "release/runtime-legal/manifest.json").read_bytes())
    full = dict(legal["platforms"][key]["full_archive"])
    full["bytes"] = full.pop("size_bytes")
    _extract_full(_archive(full, cache, ".tar.zst"), stage / "pbs")
    _extract_full(_archive(policy["llvm"][key], cache, ".tar.zst"), stage / "compiler")
    pbs = stage / "pbs/python"
    metadata = json.loads((pbs / "PYTHON.json").read_bytes())
    if sha256(pbs / "PYTHON.json") != sha256(ROOT / "release/runtime-legal" / key / "PYTHON.json"):
        raise ValueError("PBS objects differ from the reviewed runtime metadata")
    clang = stage / "compiler/llvm/bin/clang"
    env = {k: v for k, v in os.environ.items() if k not in {"CFLAGS", "CPPFLAGS", "LDFLAGS", "CXX"}}
    flags = ["-fPIC"]
    if key == "macos-arm64":
        env["SDKROOT"] = _run(["xcrun", "--show-sdk-path"], cwd=stage).strip()
        flags += ["-arch", "arm64", "-mmacosx-version-min=11.0", "-Wl,-headerpad,40"]
    # configure handles platform defaults; explicit features match PBS's recipe.
    env.update(
        CC=shlex.quote(str(clang)),
        CC_FOR_BUILD=shlex.quote(str(clang)),
        AR=shlex.quote(str(clang.with_name("llvm-ar"))),
        RANLIB=shlex.quote(str(clang.with_name("llvm-ranlib"))),
        CFLAGS=" ".join(["-O2", *flags, *("-D" + d for d in DEFINES)]),
        CPPFLAGS=" ".join(flags),
        LDFLAGS=" ".join(flags),
    )
    sql = next((stage / "sqlite").glob("sqlite-autoconf-*"))
    _run(["sh", "configure", "--disable-shared", "--prefix=/tools/deps"], cwd=sql, env=env)
    _run(["make", "-j", str(min(os.cpu_count() or 2, 8)), "libsqlite3.a"], cwd=sql, env=env)
    build = metadata["build_info"]
    config = (pbs / build["inittab_source"]).read_text()
    names = set(re.findall(r'\{"([A-Za-z0-9_]+)"\s*,', config))
    entries = [build["core"]] + [
        v
        for name, variants in build["extensions"].items()
        if name in names
        for v in variants
        if v["variant"] == "default"
    ]
    objs = list(dict.fromkeys(o for entry in entries for o in entry["objs"]))
    links: list[str | Path] = []
    seen: set[str] = set()
    for entry in entries:
        for link in entry.get("links", []):
            identity = json.dumps(link, sort_keys=True)
            if identity in seen:
                continue
            seen.add(identity)
            if link.get("path_static"):
                links.append(
                    sql / "libsqlite3.a" if link["name"] == "sqlite3" else pbs / link["path_static"]
                )
            elif link.get("framework"):
                links.extend(["-framework", link["name"]])
            else:
                links.append("-l" + link["name"])
    prefix = pbs / "install"
    if key == "macos-arm64":
        # Apple's linker preserves PBS's stack size; ld64.lld ignores that flag.
        flags += ["-flto=full", "-Wl,-lto_library," + str(clang.parent.parent / "lib/libLTO.dylib")]
        links.append(pbs / "build/lib/libclang_rt.osx.a")
        shared_flags = [
            "-dynamiclib",
            "-install_name",
            "@rpath/libpython3.13.dylib",
            "-Wl,-compatibility_version,3.13",
            "-Wl,-current_version,3.13",
        ]
        exe_flags = [
            "-Wl,-stack_size,1000000",
            "-Wl,-export_dynamic",
            "-Wl,-rpath,@executable_path/../lib",
        ]
    else:
        flags += ["-fuse-ld=lld"]
        links = ["-Wl,--start-group", *links, "-Wl,--end-group"]
        shared_flags = ["-shared", "-Wl,-soname,libpython3.13.so.1.0"]
        exe_flags = ["-Wl,-export-dynamic"]
    common = [*[pbs / obj for obj in objs], *links]
    _run(
        [clang, *flags, *shared_flags, "-o", prefix / _modified_files(key)[1], *common],
        cwd=stage,
        env=env,
    )
    _run(
        [
            clang,
            *flags,
            *exe_flags,
            "-o",
            _python(prefix),
            pbs / "build/Programs/python.o",
            *common,
        ],
        cwd=stage,
        env=env,
    )
    if key == "macos-arm64":
        for name in _modified_files(key):
            _run(["codesign", "--force", "--sign", "-", prefix / name], cwd=stage)
    return prefix, {
        "pbs_archive_sha256": full["sha256"],
        "compiler_archive_sha256": policy["llvm"][key]["sha256"],
        "macos_sdk": _run(["xcrun", "--show-sdk-version"], cwd=stage).strip()
        if key == "macos-arm64"
        else None,
    }


def _windows_build(stage: Path, cache: Path) -> tuple[Path, dict[str, Any]]:
    legal = json.loads((ROOT / "release/runtime-legal/manifest.json").read_bytes())
    pin = dict(legal["platforms"]["windows-x86_64"]["full_archive"])
    pin["bytes"] = pin.pop("size_bytes")
    _extract_full(_archive(pin, cache, ".tar.zst"), stage / "pbs")
    prefix = stage / "pbs/python/install"
    sql = next((stage / "sqlite").glob("sqlite-autoconf-*"))
    compiler = shutil.which("cl")
    if compiler is None:
        raise ValueError("SQLite runtime build requires the MSVC x64 developer environment")
    defines = [
        *DEFINES,
        "SQLITE_ENABLE_MATH_FUNCTIONS",
        "SQLITE_OMIT_AUTOINIT",
        "SQLITE_API=__declspec(dllexport)",
    ]
    _run(
        [
            compiler,
            "/nologo",
            "/O2",
            "/MD",
            "/LD",
            *["/D" + d for d in defines],
            sql / "sqlite3.c",
            "/link",
            "/OUT:" + str(prefix / "DLLs/sqlite3.dll"),
        ],
        cwd=sql,
    )
    return prefix, {
        "pbs_archive_sha256": pin["sha256"],
        "compiler": "MSVC x64 /O2 /MD",
        "compiler_sha256": sha256(Path(compiler)),
    }


def build(output: Path, *, cache: Path | None = None) -> Path:
    """Create a new prefix atomically; failed builds never become reusable runtimes."""
    if output.is_symlink():
        raise ValueError("Patched Python output must not be a symlink")
    output = output.resolve()
    if output.exists():
        raise ValueError("Patched Python output must be a new path")
    key = platform_key()
    if key not in {"macos-arm64", "windows-x86_64", "linux-x86_64"}:
        raise ValueError("No reviewed SQLite build recipe for this platform")
    policy = json.loads((ROOT / POLICY).read_bytes())
    cache = (cache or output.parent / "runtime-downloads").resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    # SQLite's configure explicitly refuses source paths containing spaces.
    # Use a private POSIX /tmp child; the completed prefix is staged separately
    # beside its destination for atomic promotion, including across filesystems.
    with tempfile.TemporaryDirectory(
        prefix="neurocraft-sqlite-", dir=None if sys.platform == "win32" else "/tmp"
    ) as temporary:
        stage = Path(temporary)
        archive = _archive(policy["source"], cache, ".tar.gz")
        if hashlib.sha3_256(archive.read_bytes()).hexdigest() != policy["source"]["sha3_256"]:
            raise ValueError("SQLite archive differs from the owning release's SHA3 digest")
        extract(archive, stage / "sqlite")
        prefix, inputs = (
            _windows_build(stage, cache)
            if sys.platform == "win32"
            else _posix_build(stage, cache, policy, key)
        )
        actual = probe(prefix)
        if actual["version"] != policy["version"] or actual["source_id"] != policy["source_id"]:
            raise ValueError("Rebuilt Python did not load the pinned patched SQLite")
        _run(
            [
                _python(prefix),
                "-I",
                "-S",
                "-B",
                "-c",
                "import _testcapi, _testbuffer, _ctypes_test",
            ],
            cwd=stage,
        )
        standard_tests = _run([_python(prefix), "-I", "-m", "test", "test_sqlite3"], cwd=stage)
        summary = re.search(r"Total tests: run=(\d+) skipped=(\d+)", standard_tests)
        if "Result: SUCCESS" not in standard_tests or summary is None:
            raise ValueError("CPython SQLite regression suite did not produce a success summary")
        # Full archives supply tests for verification, but player runtimes do not
        # need CPython's development test suite. Keep the recorded test result.
        metadata = json.loads((stage / "pbs/python/PYTHON.json").read_bytes())
        stdlib = prefix / ("Lib" if sys.platform == "win32" else "lib/python3.13")
        for package in metadata["python_stdlib_test_packages"]:
            parts = package.split(".")
            if not all(re.fullmatch(r"[A-Za-z0-9_-]+", part) for part in parts):
                raise ValueError("Unexpected upstream standard-library test package path")
            path = stdlib.joinpath(*parts)
            if path.is_dir():
                shutil.rmtree(path)
        # Match PBS's install-only distribution: linking archives are build
        # inputs, not player requirements (about 71 MiB on macOS).
        for library in prefix.rglob("libpython3.13.a"):
            library.unlink()
        receipt = {
            "schema": "neurocraft.sqlite-build.v1",
            "platform": key,
            "policy_sha256": sha256(ROOT / POLICY),
            "recipe_sha256": sha256(Path(__file__)),
            "sqlite_source_sha256": policy["source"]["sha256"],
            "inputs": inputs,
            "native_checks": actual,
            "native_extension_imports": ["_testcapi", "_testbuffer", "_ctypes_test"],
            "standard_library_tests": {
                "suite": "test_sqlite3",
                "status": "passed",
                "run": int(summary[1]),
                "skipped": int(summary[2]),
            },
            "modified_files": {name: sha256(prefix / name) for name in _modified_files(key)},
        }
        (prefix / RECEIPT).write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        verify_runtime(prefix)
        with tempfile.TemporaryDirectory(prefix=".patched-runtime-", dir=output.parent) as local:
            copied = Path(local) / "python"
            shutil.copytree(prefix, copied, symlinks=True)
            verify_runtime(copied)
            copied.rename(output)
    return output
