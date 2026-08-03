"""Local QAIRT/QNN compilation and x86 HTP-emulation runner for the injection op.

Discovery-driven: nothing is hard-coded against a specific QAIRT release.
When an SDK is present this script
  1. reads the SDK version (sdk.yaml / sdk.json in $QNN_SDK_ROOT);
  2. asks each tool for ``--help`` and only uses flags the tool actually
     advertises (e.g. context generation consumes a DLC via ``--dlc_path`` on
     current releases; older releases took a model .so via ``--model``);
  3. converts ONNX -> DLC, generates a context binary from the DLC, and runs it
     with ``qnn-net-run --retrieve_context <binary>`` — a DLC is NOT a
     directly executable QNN model;
  4. compares emulation output against local ONNX Runtime.

Without an SDK it prints an UNVALIDATED command template clearly marked as
such (we have not run these commands here) plus pointers to the docs.

Path B uses ONNX Runtime's QNN Execution Provider with
``session.disable_cpu_ep_fallback=1`` so unsupported nodes fail hard instead
of silently running on CPU.

What x86 emulation proves: graph acceptance by the QNN toolchain and
functional numerics. What it does NOT prove: Snapdragon latency, memory
pressure, power, scheduling, driver behavior, or device-specific fusions.

Usage:
    python -m steering_poc.qualcomm.qnn_emulation \
        [--model artifacts/steering_injection_prefill.onnx] [--run]
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np


def sdk_root() -> Path | None:
    root = os.environ.get("QNN_SDK_ROOT") or os.environ.get("QAIRT_SDK_ROOT")
    return Path(root) if root and Path(root).exists() else None


def sdk_version(root: Path) -> str:
    """Best-effort SDK version from sdk.yaml / sdk.json shipped at the root."""
    for name in ("sdk.yaml", "sdk.json"):
        f = root / name
        if f.exists():
            text = f.read_text(errors="replace")
            for line in text.splitlines():
                if "version" in line.lower():
                    return line.strip()
    # Fall back to the version-numbered directory convention .../qairt/<ver>
    return root.name


def find_tool(name: str):
    hit = shutil.which(name) or shutil.which(name + ".exe")
    if hit:
        return hit
    root = sdk_root()
    if root:
        hits = sorted(root.glob(f"bin/*/{name}*"))
        if hits:
            return str(hits[0])
    return None


def tool_help(tool_path: str) -> str:
    """Return the tool's --help text (used to discover supported flags)."""
    try:
        res = subprocess.run([tool_path, "--help"], capture_output=True,
                             text=True, timeout=60)
        return (res.stdout or "") + (res.stderr or "")
    except Exception as e:
        return f"<--help failed: {e}>"


def supports_flag(help_text: str, flag: str) -> bool:
    return flag in help_text


def _write_reference_inputs(model_path: Path, out_dir: Path):
    import onnx

    m = onnx.load(str(model_path))
    rng = np.random.default_rng(0)
    entries, arrays = [], {}
    for inp in m.graph.input:
        dims = [d.dim_value or 1 for d in inp.type.tensor_type.shape.dim] or [1]
        arr = (np.array([2.0], np.float32) if inp.name == "alpha"
               else np.ones(dims, np.float32) if inp.name == "mask"
               else rng.standard_normal(dims).astype(np.float32))
        raw = out_dir / f"{inp.name}.raw"
        arr.tofile(raw)
        entries.append(f"{inp.name}:={raw.name}")
        arrays[inp.name] = arr
    (out_dir / "input_list.txt").write_text(" ".join(entries) + "\n")
    ref = arrays["hidden"] + arrays["alpha"] * arrays["mask"] * arrays["steering"]
    return ref


def try_ort_qnn(model_path: Path, fail_hard: bool = True) -> bool:
    import onnxruntime as ort

    if "QNNExecutionProvider" not in ort.get_available_providers():
        print("ONNX Runtime QNN EP not available in this build "
              "(pip install onnxruntime-qnn on a supported platform).")
        return False

    so = ort.SessionOptions()
    if fail_hard:
        so.add_session_config_entry("session.disable_cpu_ep_fallback", "1")

    sess = ort.InferenceSession(
        str(model_path),
        sess_options=so,
        providers=["QNNExecutionProvider"],
        provider_options=[{"backend_path": "QnnHtp.dll"
                           if sys.platform == "win32" else "libQnnHtp.so"}],
    )
    rng = np.random.default_rng(0)
    feeds = {}
    for inp in sess.get_inputs():
        dims = [d if isinstance(d, int) else 1 for d in inp.shape]
        feeds[inp.name] = (
            np.array([2.0], dtype=np.float32) if inp.name == "alpha"
            else np.ones(dims, np.float32) if inp.name == "mask"
            else rng.standard_normal(dims).astype(np.float32)
        )
    out = sess.run(None, feeds)[0]
    ref = feeds["hidden"] + feeds["alpha"] * feeds["mask"] * feeds["steering"]
    err = float(np.abs(out - ref).max())
    print(f"QNN EP ran the graph (CPU fallback disabled). "
          f"max_abs_err vs numpy reference: {err:.3e}")
    return True


def _run_logged(cmd: list[str], cwd=None) -> bool:
    print("Running:", " ".join(str(c) for c in cmd))
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)
    if res.stdout:
        print(res.stdout[-1500:])
    if res.returncode != 0:
        print(res.stderr[-1500:], file=sys.stderr)
        return False
    return True


def qairt_cli_pipeline(model_path: Path, out_dir: Path, run: bool) -> bool:
    root = sdk_root()
    converter = find_tool("qairt-converter") or find_tool("qnn-onnx-converter")

    if not converter:
        print("No QAIRT SDK found (set QNN_SDK_ROOT).")
        print("""
UNVALIDATED template — we have NOT executed these commands in this repo; the
flags below were current in QAIRT docs at authoring time and MUST be checked
against `<tool> --help` of your installed SDK before use:

  1. ONNX -> DLC             qairt-converter --input_network <model.onnx> \\
                                             --output_path steering.dlc
  2. DLC -> context binary   qnn-context-binary-generator \\
                                 --backend libQnnHtp.so --dlc_path steering.dlc \\
                                 --binary_file steering_ctx
  3. Run the CONTEXT (not the DLC) under x86 HTP emulation:
                             qnn-net-run --retrieve_context steering_ctx.bin \\
                                 --backend libQnnHtp.so --input_list input_list.txt

Docs: https://docs.qualcomm.com/bundle/publicresource/topics/80-63442-50/
""")
        return False

    print(f"QAIRT SDK: {root} (version: {sdk_version(root) if root else 'unknown'})")
    print(f"Converter: {converter}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. ONNX -> DLC (flags verified against the tool's own --help) ------
    conv_help = tool_help(converter)
    dlc = out_dir / "steering_injection.dlc"
    if supports_flag(conv_help, "--input_network"):
        cmd = [converter, "--input_network", str(model_path),
               "--output_path", str(dlc)]
    elif supports_flag(conv_help, "--input_model"):
        cmd = [converter, "--input_model", str(model_path),
               "--output_path", str(dlc)]
    else:
        print("Converter --help exposes neither --input_network nor "
              "--input_model; inspect it manually:")
        print(conv_help[:1200])
        return False
    if not _run_logged(cmd):
        return False
    print(f"Converted -> {dlc}")

    # --- 2. DLC -> context binary ------------------------------------------
    ctxgen = find_tool("qnn-context-binary-generator")
    ctx_path = None
    if ctxgen:
        gen_help = tool_help(ctxgen)
        backend = None
        if root:
            backend = next(iter(root.glob("lib/*/libQnnHtp.so")), None) or \
                next(iter(root.glob("lib/*/QnnHtp.dll")), None)
        cmd = [ctxgen, "--binary_file", "steering_ctx"]
        if supports_flag(gen_help, "--dlc_path"):
            cmd += ["--dlc_path", str(dlc)]
        elif supports_flag(gen_help, "--model"):
            # Legacy route: needs a model .so from qnn-model-lib-generator first.
            print("This SDK's context generator wants --model (model library), "
                  "not --dlc_path; run qnn-model-lib-generator first. Skipping.")
            cmd = None
        if cmd is not None and backend:
            cmd += ["--backend", str(backend)]
        if cmd is not None and _run_logged(cmd, cwd=out_dir):
            hits = list(out_dir.glob("**/steering_ctx*.bin")) or \
                list(out_dir.glob("**/steering_ctx*"))
            ctx_path = hits[0] if hits else None
            print(f"Context binary -> {ctx_path}")
    else:
        print("qnn-context-binary-generator not found; stopping after DLC.")

    # --- 3. Run the context under x86 HTP emulation ------------------------
    if run and ctx_path:
        net_run = find_tool("qnn-net-run")
        if not net_run:
            print("qnn-net-run not found; context generated but not executed.")
            return True
        run_help = tool_help(net_run)
        if not supports_flag(run_help, "--retrieve_context"):
            print("qnn-net-run --help does not list --retrieve_context; "
                  "inspect manually before running.")
            return True
        ref = _write_reference_inputs(model_path, out_dir)
        backend = next(iter(root.glob("lib/*/libQnnHtp.so")), None) if root else None
        cmd = [net_run, "--retrieve_context", str(ctx_path),
               "--input_list", str(out_dir / "input_list.txt")]
        if backend:
            cmd += ["--backend", str(backend)]
        if not _run_logged(cmd, cwd=out_dir):
            return False
        out_files = sorted(out_dir.glob("output/Result_0/*.raw"))
        if out_files:
            got = np.fromfile(out_files[0], dtype=np.float32).reshape(ref.shape)
            err = float(np.abs(got - ref).max())
            print(f"Emulation vs numpy reference: max_abs_err={err:.3e}")
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model",
                        default="artifacts/steering_injection_prefill.onnx")
    parser.add_argument("--out-dir", default="artifacts/qnn")
    parser.add_argument("--run", action="store_true",
                        help="Also execute the context via x86 HTP emulation")
    args = parser.parse_args(argv)

    model_path = Path(args.model)
    if not model_path.exists():
        raise SystemExit(f"{model_path} not found — run steering_poc.export_onnx")

    print("--- Path B: ONNX Runtime QNN EP ---")
    ort_ok = try_ort_qnn(model_path)
    print("\n--- Path A: QAIRT SDK CLI ---")
    cli_ok = qairt_cli_pipeline(model_path, Path(args.out_dir), args.run)

    if not (ort_ok or cli_ok):
        print("\nNo local Qualcomm toolchain available. Use the AI Hub cloud "
              "path instead: python -m steering_poc.qualcomm.submit_ai_hub")


if __name__ == "__main__":
    main()
