"""Detect available Qualcomm tooling WITHOUT requiring Qualcomm hardware.

Checks (never prints secrets):
  * qai-hub / qai-hub-models Python packages
  * AI Hub API token configured (~/.qai_hub/client.ini or QAI_HUB_API_TOKEN set)
  * onnxruntime QNN Execution Provider
  * QAIRT / QNN SDK (QNN_SDK_ROOT / QAIRT_SDK_ROOT env vars, converter binaries)
  * ExecuTorch with Qualcomm backend

Usage:
    python -m steering_poc.qualcomm.detect_environment
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
from pathlib import Path


def _pkg(name: str):
    spec = importlib.util.find_spec(name)
    if spec is None:
        return {"available": False}
    try:
        mod = __import__(name)
        return {"available": True, "version": getattr(mod, "__version__", "unknown")}
    except Exception as e:
        return {"available": False, "error": f"import failed: {type(e).__name__}"}


def detect() -> dict:
    report: dict = {}

    report["qai_hub"] = _pkg("qai_hub")
    report["qai_hub_models"] = _pkg("qai_hub_models")

    # AI Hub token: presence only — value is never read into the report.
    token_file = Path.home() / ".qai_hub" / "client.ini"
    report["ai_hub_token_configured"] = bool(
        token_file.exists() or os.environ.get("QAI_HUB_API_TOKEN")
    )

    ort = _pkg("onnxruntime")
    report["onnxruntime"] = ort
    if ort.get("available"):
        import onnxruntime

        providers = onnxruntime.get_available_providers()
        report["onnxruntime"]["providers"] = providers
        report["onnxruntime"]["qnn_ep"] = "QNNExecutionProvider" in providers

    sdk_root = os.environ.get("QNN_SDK_ROOT") or os.environ.get("QAIRT_SDK_ROOT")
    report["qairt_sdk"] = {
        "env_root": sdk_root if sdk_root else None,
        "root_exists": bool(sdk_root and Path(sdk_root).exists()),
    }
    for tool in (
        "qnn-onnx-converter",
        "qairt-converter",
        "qnn-context-binary-generator",
        "qnn-net-run",
    ):
        found = shutil.which(tool) or shutil.which(tool + ".exe")
        if not found and sdk_root:
            hits = list(Path(sdk_root).glob(f"bin/*/{tool}*"))
            found = str(hits[0]) if hits else None
        report["qairt_sdk"][tool] = found

    et = _pkg("executorch")
    report["executorch"] = et
    if et.get("available"):
        report["executorch"]["qualcomm_backend"] = (
            importlib.util.find_spec("executorch.backends.qualcomm") is not None
        )

    return report


def summarize(report: dict) -> list[str]:
    lines = []

    def yn(flag):
        return "YES" if flag else "no"

    lines.append(f"qai-hub package:          {yn(report['qai_hub']['available'])}"
                 + (f" (v{report['qai_hub'].get('version')})"
                    if report["qai_hub"]["available"] else ""))
    lines.append(f"qai-hub-models package:   {yn(report['qai_hub_models']['available'])}")
    lines.append(f"AI Hub token configured:  {yn(report['ai_hub_token_configured'])}"
                 " (value not shown)")
    ort = report["onnxruntime"]
    lines.append(f"onnxruntime:              {yn(ort.get('available'))}"
                 + (f" (v{ort.get('version')}, QNN EP: {yn(ort.get('qnn_ep'))})"
                    if ort.get("available") else ""))
    sdk = report["qairt_sdk"]
    lines.append(f"QAIRT/QNN SDK root:       {sdk['env_root'] or 'not set'}"
                 + (" [exists]" if sdk["root_exists"] else ""))
    for tool in ("qnn-onnx-converter", "qairt-converter",
                 "qnn-context-binary-generator", "qnn-net-run"):
        lines.append(f"  {tool:<28}{sdk.get(tool) or 'not found'}")
    et = report["executorch"]
    lines.append(f"executorch:               {yn(et.get('available'))}"
                 + (f" (qualcomm backend: {yn(et.get('qualcomm_backend'))})"
                    if et.get("available") else ""))
    return lines


def main():
    report = detect()
    print("Qualcomm environment detection")
    print("=" * 60)
    for line in summarize(report):
        print(line)
    out = Path("artifacts/qualcomm_environment.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved -> {out}")

    print("\nNext steps:")
    if not report["qai_hub"]["available"]:
        print("  * pip install qai-hub qai-hub-models")
    if not report["ai_hub_token_configured"]:
        print("  * qai-hub configure --api_token <token from "
              "https://workbench.aihub.qualcomm.com/account/> (do not commit it)")
    if report["qai_hub"]["available"] and report["ai_hub_token_configured"]:
        print("  * python -m steering_poc.qualcomm.submit_ai_hub --list-devices")
        print("  * python -m steering_poc.qualcomm.submit_ai_hub --submit ...")


if __name__ == "__main__":
    main()
