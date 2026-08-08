#!/usr/bin/env python3
"""Render flattened xutf and simdutf transcoding assembly.

Compiles an inspection wrapper for xutf and simdutf's C bridge, then prints the
selected function bodies for the requested conversion. The default output
covers the local CPU and an x86-64 Sapphire Rapids target.

Examples:
    uv run scripts/asm.py utf16-to-utf8 | less
    uv run scripts/asm.py utf8-to-utf32 --target sapphire-rapids --output asm.s

The staged x86 dispatch is folded to the selected CPU: compatible
`target_feature` barriers become ``#[inline(always)]`` and unsupported paths
are removed. Production sources remain untouched; the tool rejects an entry
that still directly calls xutf code. The simdutf entry is the checked
``convert_*`` call used by the throughput summary, followed by the
implementation selected for that target.
"""

from __future__ import annotations

from collections.abc import Callable

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Operation:
    """One benchmarked transcoding pair and its inspection symbols."""

    name: str
    title: str
    xutf_symbol: str
    simdutf_symbol: str
    simdutf_method: str


OPERATIONS = {
    operation.name: operation
    for operation in (
        Operation(
            "utf8-to-utf16",
            "UTF-8 → UTF-16",
            "xutf_utf8_to_utf16",
            "simdutf_convert_utf8_to_utf16",
            "convert_utf8_to_utf16le",
        ),
        Operation(
            "utf16-to-utf8",
            "UTF-16 → UTF-8",
            "xutf_utf16_to_utf8",
            "simdutf_convert_utf16_to_utf8",
            "convert_utf16le_to_utf8",
        ),
        Operation(
            "utf8-to-utf32",
            "UTF-8 → UTF-32",
            "xutf_utf8_to_utf32",
            "simdutf_convert_utf8_to_utf32",
            "convert_utf8_to_utf32",
        ),
        Operation(
            "utf32-to-utf8",
            "UTF-32 → UTF-8",
            "xutf_utf32_to_utf8",
            "simdutf_convert_utf32_to_utf8",
            "convert_utf32_to_utf8",
        ),
    )
}
LABEL = re.compile(r"^\s*([.$A-Za-z_][.$A-Za-z0-9_@]*)\s*:")
TARGET_FEATURE = re.compile(
    r'^(?P<indent>[ \t]*)#\[target_feature\(enable = "(?P<features>[^"]+)"\)\]$',
    re.MULTILINE,
)
DIRECT_CALL = re.compile(r"^\s*(?:callq?|bl)\s+([.$A-Za-z_][.$A-Za-z0-9_@]*)")
INLINE_HINT = re.compile(r"^(?P<indent>[ \t]*)#\[inline\]$", re.MULTILINE)
INLINE_ALWAYS = re.compile(r"^\s*#\[inline\(always\)\]\s*$")
INLINE_ATTRIBUTE = re.compile(r"^\s*#\[inline(?:\([^)]*\))?\]\s*$")
INLINE_NEVER = re.compile(r"^(?P<indent>[ \t]*)#\[inline\(never\)\]$", re.MULTILINE)
FUNCTION_START = re.compile(
    r"^(?P<indent>[ \t]*)(?:pub(?:\([^)]*\))?\s+)?(?:(?:const|unsafe|async)\s+)*"
    r'(?:extern(?:\s+"[^"]+")?\s+)?fn\s+[A-Za-z_]\w*'
)
X86_FEATURE_GATE = re.compile(
    r"(?ms)^(?P<indent>[ \t]*)fn "
    r"(?P<name>has_ssse3|has_avx512_vbmi2)\(\) -> bool \{\n.*?^(?P=indent)\}"
)
X86_FEATURE_MACROS = {
    "__SSSE3__": "ssse3",
    "__SSE4_2__": "sse4.2",
    "__XSAVE__": "xsave",
    "__BMI2__": "bmi2",
    "__AVX2__": "avx2",
    "__AVX512F__": "avx512f",
    "__AVX512DQ__": "avx512dq",
    "__AVX512VL__": "avx512vl",
    "__AVX512BW__": "avx512bw",
    "__AVX512VBMI__": "avx512vbmi",
    "__AVX512VBMI2__": "avx512vbmi2",
}
SIMDUTF_ICELAKE_FEATURES = frozenset(
    {"avx2", "avx512f", "avx512dq", "avx512vl", "avx512vbmi2", "bmi2"}
)
XUTF_AVX512_FEATURES = frozenset(
    {"avx2", "avx512f", "avx512bw", "avx512vbmi", "avx512vbmi2", "bmi2"}
)
SAPPHIRE_RAPIDS_FEATURES = frozenset(X86_FEATURE_MACROS.values())


class ToolError(RuntimeError):
    """Describes an unavailable compiler, target, or expected assembly symbol."""


def command_text(command: list[str]) -> str:
    """Formats a subprocess command for an actionable failure message."""

    return " ".join(shlex.quote(part) for part in command)


def run(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
    input_text: str | None = None,
) -> subprocess.CompletedProcess[str]:
    """Runs a compiler command and preserves its diagnostics on failure."""

    result = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        input=input_text,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode == 0:
        return result

    output = "\n".join(
        part for part in (result.stdout.strip(), result.stderr.strip()) if part
    )
    raise ToolError(f"{command_text(command)} failed\n{output}")


def project_root() -> Path:
    """Finds the repository root from this script's stable location."""

    return Path(__file__).resolve().parent.parent


def cargo_metadata(root: Path) -> dict[str, object]:
    """Loads resolved package metadata without changing Cargo.lock."""

    result = run(["cargo", "metadata", "--locked", "--format-version=1"], cwd=root)
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise ToolError(f"cargo metadata returned invalid JSON: {error}") from error


def simdutf_source(root: Path) -> tuple[Path, str]:
    """Locates the exact simdutf C++ source locked for the benchmark."""

    def locate(metadata: dict[str, object]) -> tuple[Path, str] | None:
        packages = metadata.get("packages")
        if not isinstance(packages, list):
            raise ToolError("cargo metadata did not contain a package list")
        matches = [package for package in packages if package.get("name") == "simdutf"]
        if len(matches) != 1:
            raise ToolError("expected exactly one resolved simdutf package")
        package = matches[0]
        manifest = package.get("manifest_path")
        version = package.get("version")
        if not isinstance(manifest, str) or not isinstance(version, str):
            raise ToolError("simdutf metadata lacks its manifest path or version")
        source = Path(manifest).parent / "cpp" / "simdutfrs.cpp"
        return (source, version) if source.is_file() else None

    metadata = cargo_metadata(root)
    found = locate(metadata)
    if found is not None:
        return found

    run(["cargo", "fetch", "--locked"], cwd=root)
    found = locate(cargo_metadata(root))
    if found is None:
        raise ToolError("simdutf source is unavailable after cargo fetch")
    return found


def rust_host(root: Path) -> str:
    """Returns Rust's host target triple."""

    result = run(["rustc", "-vV"], cwd=root)
    for line in result.stdout.splitlines():
        if line.startswith("host: "):
            return line.removeprefix("host: ")
    raise ToolError("rustc -vV did not report a host target")


def cxx_command() -> list[str]:
    """Resolves the C++ compiler while honoring the conventional CXX override."""

    command = shlex.split(os.environ.get("CXX", "clang++"))
    if not command or shutil.which(command[0]) is None:
        raise ToolError(f"C++ compiler not found: {command[0] if command else 'CXX'}")
    return command


def native_x86_features(root: Path, host: str, cxx: list[str]) -> frozenset[str]:
    """Reads the native x86 feature closure shared by both assembly builds."""

    if not host.startswith("x86_64-"):
        return frozenset()
    result = run(
        [*cxx, "-dM", "-E", "-x", "c++", "-march=native", "-"],
        cwd=root,
        input_text="",
    )
    macros = {
        line.split()[1]
        for line in result.stdout.splitlines()
        if line.startswith("#define ")
    }
    return frozenset(
        feature for macro, feature in X86_FEATURE_MACROS.items() if macro in macros
    )


def native_implementation(host: str, features: frozenset[str]) -> str:
    """Identifies simdutf's implementation from the native CPU feature closure."""

    if host.startswith("aarch64-"):
        return "arm64"
    if not host.startswith("x86_64-"):
        raise ToolError(f"native simdutf extraction does not support {host}")
    if SIMDUTF_ICELAKE_FEATURES <= features:
        return "icelake"
    if "avx2" in features:
        return "haswell"
    if "sse4.2" in features:
        return "westmere"
    return "fallback"


@dataclass(frozen=True)
class Target:
    """A Rust/C++ CPU configuration, x86 feature closure, and simdutf selection."""

    name: str
    label: str
    rust_target: str
    rust_cpu: str
    cxx_flags: tuple[str, ...]
    simdutf_implementation: str
    x86_features: frozenset[str]


def sapphire_target(host: str) -> tuple[str, tuple[str, ...]]:
    """Maps the host OS to a compilable Sapphire Rapids Rust/C++ target."""

    if host.endswith("apple-darwin"):
        return ("x86_64-apple-darwin", ("-arch", "x86_64", "-march=sapphirerapids"))
    if "linux" in host:
        return (
            "x86_64-unknown-linux-gnu",
            ("--target=x86_64-unknown-linux-gnu", "-march=sapphirerapids"),
        )
    raise ToolError(f"Sapphire Rapids assembly is unsupported from host target {host}")


def targets(root: Path, requested: str, host: str, cxx: list[str]) -> list[Target]:
    """Builds the requested target set in stable display order."""

    chosen: list[Target] = []
    if requested in ("all", "sapphire-rapids"):
        target, cxx_flags = sapphire_target(host)
        chosen.append(
            Target(
                "sapphire-rapids",
                "x86-64 / Sapphire Rapids",
                target,
                "sapphirerapids",
                cxx_flags,
                "icelake",
                SAPPHIRE_RAPIDS_FEATURES,
            )
        )
    if requested in ("all", "native"):
        native_features = native_x86_features(root, host, cxx)
        chosen.append(
            Target(
                "native",
                f"native / {host}",
                host,
                "native",
                ("-march=native",),
                native_implementation(host, native_features),
                native_features,
            )
        )
    return chosen


def has_function_body(lines: list[str], start: int) -> bool:
    """Distinguishes Rust function definitions from declarations."""

    for line in lines[start:]:
        code = line.split("//", 1)[0]
        body = code.find("{")
        declaration = code.find(";")
        if body >= 0 and (declaration < 0 or body < declaration):
            return True
        if declaration >= 0:
            return False
    return False


def has_target_feature_attribute(lines: list[str]) -> bool:
    """Checks the contiguous attribute group before a function definition."""

    for line in reversed(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith("//"):
            continue
        if not stripped.startswith("#["):
            return False
        if stripped.startswith("#[target_feature"):
            return True
    return False


def specialize_x86_dispatch(text: str, features: frozenset[str]) -> str:
    """Folds xutf's x86 runtime gates to the inspection target's feature set."""

    def replace_gate(match: re.Match[str]) -> str:
        name = match.group("name")
        enabled = (
            "ssse3" in features
            if name == "has_ssse3"
            else XUTF_AVX512_FEATURES <= features
        )
        indent = match.group("indent")
        return f"{indent}fn {name}() -> bool {{\n{indent}    {'true' if enabled else 'false'}\n{indent}}}"

    specialized, count = X86_FEATURE_GATE.subn(replace_gate, text)
    if count != 2:
        raise ToolError(
            "xutf x86 dispatch no longer contains both expected feature gates"
        )
    return specialized


def inline_supported_target_features(text: str, features: frozenset[str]) -> str:
    """Replaces only target-feature barriers guaranteed by the inspection CPU."""

    def replace_attribute(match: re.Match[str]) -> str:
        required = frozenset(match.group("features").split(","))
        if required <= features:
            return f"{match.group('indent')}#[inline(always)]"
        return match.group(0)

    return TARGET_FEATURE.sub(replace_attribute, text)


def flatten_source(source: Path, target: Target) -> int:
    """Adds always-inline hints to every staged Rust function with a body."""

    text = source.read_text(encoding="utf-8")
    if source.name == "x86.rs" and target.rust_target.startswith("x86_64-"):
        text = specialize_x86_dispatch(text, target.x86_features)
        text = inline_supported_target_features(text, target.x86_features)
    text, _ = INLINE_NEVER.subn(r"\g<indent>#[inline(always)]", text)
    text, _ = INLINE_HINT.subn(r"\g<indent>#[inline(always)]", text)
    lines = text.splitlines(keepends=True)
    newline = "\r\n" if "\r\n" in text else "\n"
    flattened: list[str] = []
    added = 0
    for index, line in enumerate(lines):
        function = FUNCTION_START.match(line)
        if (
            function is not None
            and has_function_body(lines, index)
            and not has_target_feature_attribute(flattened)
            and (not flattened or INLINE_ATTRIBUTE.fullmatch(flattened[-1]) is None)
        ):
            flattened.append(f"{function.group('indent')}#[inline(always)]{newline}")
            added += 1
        flattened.append(line)
    source.write_text("".join(flattened), encoding="utf-8")
    return added


def staged_xutf(root: Path, workspace: Path, target: Target) -> Path:
    """Copies xutf and always-inlines its internal Rust implementation."""

    project = workspace / "source"
    project.mkdir(parents=True)
    for name in ("Cargo.toml", "Cargo.lock", "rust-toolchain.toml", "README.md"):
        source = root / name
        if source.is_file():
            shutil.copy2(source, project / name)
    shutil.copytree(root / "src", project / "src")
    shutil.copytree(root / "benches", project / "benches")

    added = 0
    for source in sorted((project / "src").rglob("*.rs")):
        if source.parent.name != "bin":
            added += flatten_source(source, target)
    if added == 0:
        raise ToolError("xutf source contains no internal functions to flatten")
    return project


def compile_xutf(
    root: Path, workspace: Path, target: Target, operation: Operation
) -> Path:
    """Emits the inspection fixture's recursively flattened Rust entry assembly."""

    project = staged_xutf(root, workspace, target)
    target_dir = workspace / "target"
    env = dict(os.environ)
    env.pop("RUSTFLAGS", None)
    env.pop("CARGO_ENCODED_RUSTFLAGS", None)
    target_env = (
        f"CARGO_TARGET_{target.rust_target.upper().replace('-', '_')}_RUSTFLAGS"
    )
    env |= {
        "CARGO_TARGET_DIR": str(target_dir),
        "CARGO_TERM_COLOR": "never",
        target_env: f"-C target-cpu={target.rust_cpu}",
    }
    run(
        [
            "cargo",
            "rustc",
            "--locked",
            "--release",
            "--bin",
            "asm_fixture",
            "--target",
            target.rust_target,
            "--",
            "-C",
            f"target-cpu={target.rust_cpu}",
            "--emit=asm",
        ],
        cwd=project,
        env=env,
    )
    assembly = sorted(target_dir.rglob("*.s"))
    entries = [
        path
        for path in assembly
        if operation.xutf_symbol in path.read_text(encoding="utf-8")
    ]
    if len(entries) != 1:
        paths = ", ".join(str(path) for path in entries) or "none"
        raise ToolError(
            "cargo did not emit exactly one assembly entry for "
            f"src/bin/asm_fixture.rs (matches: {paths})"
        )
    return entries[0]


def compile_simdutf(
    root: Path, source: Path, output: Path, target: Target, cxx: list[str]
) -> None:
    """Emits optimized assembly for the locked simdutf C bridge."""

    run(
        [
            *cxx,
            "-O3",
            "-DNDEBUG",
            "-std=c++11",
            "-S",
            *target.cxx_flags,
            "-o",
            str(output),
            str(source),
        ],
        cwd=root,
    )


def label_at(line: str) -> str | None:
    """Returns an assembler label from one source line, if present."""

    match = LABEL.match(line)
    return match.group(1) if match else None


def function_end(lines: list[str], label_index: int, description: str) -> int:
    """Finds a compiler-recognized end marker for one function body."""

    for index in range(label_index, len(lines)):
        if "-- End function" in lines[index]:
            return index
        if ".cfi_endproc" in lines[index]:
            if index + 1 < len(lines) and "-- End function" in lines[index + 1]:
                return index + 1
            return index
    raise ToolError(f"assembly ended before {description} finished")


def rust_constant_start(lines: list[str], label_index: int, end: int) -> int:
    """Finds the constant pool LLVM attached to a Rust function, if present."""

    block = next(
        (
            re.search(r"\bLBB(\d+)_", line)
            for line in lines[label_index : end + 1]
            if "LBB" in line
        ),
        None,
    )
    if block is None:
        return label_index

    prefix = f"LCPI{block.group(1)}_"
    constants = [
        index for index, line in enumerate(lines[:label_index]) if prefix in line
    ]
    if not constants:
        return label_index

    for index in range(constants[0], -1, -1):
        if lines[index].lstrip().startswith(".section"):
            return index
    return label_index


def function_region(lines: list[str], label_index: int, description: str) -> str:
    """Keeps a function and its immediately preceding constant pool."""

    end = function_end(lines, label_index, description)
    previous = next(
        (
            index
            for index in range(label_index - 1, -1, -1)
            if "-- End function" in lines[index]
        ),
        None,
    )
    start = (
        previous + 1
        if previous is not None
        else rust_constant_start(lines, label_index, end)
    )
    return "\n".join(lines[start : end + 1]).strip()


def extract_function(
    assembly: Path, description: str, predicate: Callable[[str], bool]
) -> str:
    """Extracts one compiler-delimited function region from an assembly file."""

    lines = assembly.read_text(encoding="utf-8").splitlines()
    for index, line in enumerate(lines):
        label = label_at(line)
        if label is not None and predicate(label):
            return function_region(lines, index, description)
    raise ToolError(f"could not find {description} in {assembly}")


def flattened_xutf_entry(assembly: Path, operation: Operation) -> str:
    """Extracts a fully inlined xutf entry or names the remaining call barrier."""

    entry = extract_function(
        assembly,
        f"xutf {operation.xutf_symbol}",
        exported_symbol(operation.xutf_symbol),
    )
    callees = sorted(
        {
            match.group(1).split("@", 1)[0]
            for line in entry.splitlines()
            if (match := DIRECT_CALL.match(line)) is not None
            and "xutf" in match.group(1)
        }
    )
    if callees:
        names = ", ".join(callees)
        raise ToolError(
            f"flattened {operation.xutf_symbol} still calls xutf code: {names}"
        )
    return entry


def exported_symbol(symbol: str) -> Callable[[str], bool]:
    """Matches an exported C symbol on ELF and Mach-O assembly output."""

    return lambda label: label == symbol or label == f"_{symbol}"


def simdutf_kernel(implementation: str, method: str) -> Callable[[str], bool]:
    """Matches the selected simdutf C++ method without matching cold clones."""

    needle = (
        f"{len(implementation)}{implementation}14implementation{len(method)}{method}"
    )
    return lambda label: needle in label and ".cold" not in label


def render_target(
    operation: Operation,
    target: Target,
    xutf_assembly: Path,
    simdutf_assembly: Path,
    simdutf_version: str,
) -> str:
    """Renders directly comparable xutf, simdutf entry, and kernel sections."""

    xutf = flattened_xutf_entry(xutf_assembly, operation)
    simdutf_entry = extract_function(
        simdutf_assembly,
        f"simdutf {operation.simdutf_symbol}",
        exported_symbol(operation.simdutf_symbol),
    )
    simdutf_body = extract_function(
        simdutf_assembly,
        f"simdutf {target.simdutf_implementation} {operation.simdutf_method}",
        simdutf_kernel(target.simdutf_implementation, operation.simdutf_method),
    )
    divider = "=" * 88
    return "\n".join(
        (
            divider,
            f"{target.label}: {operation.title}",
            f"rustc: --target {target.rust_target} -C target-cpu={target.rust_cpu}",
            f"simdutf {simdutf_version}: {' '.join(target.cxx_flags)}",
            divider,
            "",
            f"xutf flattened entry: {operation.xutf_symbol}",
            "-" * 88,
            xutf,
            "",
            f"simdutf benchmark entry: {operation.simdutf_symbol}",
            "-" * 88,
            simdutf_entry,
            "",
            f"simdutf {target.simdutf_implementation} kernel: {operation.simdutf_method}",
            "-" * 88,
            simdutf_body,
            "",
        )
    )


def parse_operation(value: str) -> Operation:
    """Accepts conventional hyphen placements while preserving canonical names."""

    normalized = value.lower().replace("-", "")
    for name, operation in OPERATIONS.items():
        if name.replace("-", "") == normalized:
            return operation
    choices = ", ".join(OPERATIONS)
    raise argparse.ArgumentTypeError(
        f"unknown operation {value!r}; choose one of {choices}"
    )


def arguments() -> argparse.Namespace:
    """Parses the operation and optional output selection."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "operation", type=parse_operation, help="conversion, e.g. utf16-to-utf8"
    )
    parser.add_argument(
        "--target",
        choices=("all", "sapphire-rapids", "native"),
        default="all",
        help="CPU configuration to render (default: all)",
    )
    parser.add_argument(
        "--output", type=Path, help="write the rendered assembly to this file"
    )
    return parser.parse_args()


def main() -> int:
    """Compiles both implementations and writes the requested assembly report."""

    args = arguments()
    root = project_root()
    try:
        cxx = cxx_command()
        host = rust_host(root)
        simdutf, version = simdutf_source(root)
        chosen_targets = targets(root, args.target, host, cxx)
        with tempfile.TemporaryDirectory(prefix="xutf-asm-") as temporary:
            workspace = Path(temporary)
            report = []
            for target in chosen_targets:
                xutf_assembly = compile_xutf(
                    root, workspace / target.name, target, args.operation
                )
                simdutf_assembly = workspace / f"simdutf-{target.name}.s"
                compile_simdutf(root, simdutf, simdutf_assembly, target, cxx)
                report.append(
                    render_target(
                        args.operation, target, xutf_assembly, simdutf_assembly, version
                    )
                )
        rendered = "\n".join(report)
    except ToolError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    if args.output is None:
        print(rendered, end="")
    else:
        args.output.write_text(rendered, encoding="utf-8")
        print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
