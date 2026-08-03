"""Opt-in Qualcomm AI Hub Workbench submission for the standalone injection op.

Tested against qai-hub == 0.52.0 and 0.53.0.
Uses the current API: ``qai_hub.Client()`` and
``submit_compile_and_link_jobs()`` (which compiles with
``--target_runtime qnn_dlc`` and links the DLC(s) into a QNN context binary —
the older direct ``--target_runtime qnn_context_binary`` compile route is
deprecated). Inference/profile jobs run against the **LinkJob's** target model.

NOTHING is submitted without --submit. Dry-run (default) prints exactly what
would be sent. Requires a configured API token
(``qai-hub configure --api_token ...``); the token is never printed.

Steps when --submit is given:
  1. resolve the target device from the LIVE device list (--device substring,
     default "Snapdragon X Elite")
  2. submit_compile_and_link_jobs on the injection ONNX
     (default: single prefill graph; --multi-graph links prefill+decode into
     one weight-shared context binary from the dynamic-shape ONNX)
  3. optionally (--infer) run on-device inference on the linked model and
     compare against local ONNX Runtime within --tol
  4. optionally (--profile) submit a profiling job on the linked model
  5. record job IDs/URLs to artifacts/ai_hub_jobs.json

Usage:
    python -m steering_poc.qualcomm.submit_ai_hub --list-devices
    python -m steering_poc.qualcomm.submit_ai_hub --submit --infer [--multi-graph]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from ..common import save_json

TESTED_QAI_HUB_VERSIONS = {"0.52.0", "0.53.0"}


def _require_client():
    try:
        import qai_hub
    except ImportError:
        print("qai-hub is not installed. Run: pip install -r requirements-qualcomm.txt\n"
              "Then: qai-hub configure --api_token <token>", file=sys.stderr)
        sys.exit(2)
    if str(qai_hub.__version__) not in TESTED_QAI_HUB_VERSIONS:
        print(f"WARNING: qai-hub {qai_hub.__version__} != tested "
              f"{sorted(TESTED_QAI_HUB_VERSIONS)}; API surface may differ.", file=sys.stderr)
    return qai_hub


def _reference_inputs(model_path: Path, seq_len: int):
    rng = np.random.default_rng(1234)
    import onnx

    m = onnx.load(str(model_path))
    data = {}
    for inp in m.graph.input:
        dims = [d.dim_value if d.HasField("dim_value") and d.dim_value > 0
                else seq_len
                for d in inp.type.tensor_type.shape.dim]
        if inp.name == "alpha":
            # Shape (1,), not rank-0: qai-hub's h5 dataset writer cannot
            # gzip scalar datasets ("Scalar datasets don't support
            # chunk/filter options"); QNN treats the input as [1] anyway.
            data[inp.name] = [np.array([2.0], dtype=np.float32)]
        elif inp.name == "mask":
            data[inp.name] = [np.ones(dims or [1], dtype=np.float32)]
        else:
            data[inp.name] = [rng.standard_normal(dims or [1]).astype(np.float32)]
    return data


def _local_ort_reference(model_path: Path, inputs: dict):
    import onnxruntime as ort

    sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    expected_rank = {i.name: len(i.shape) for i in sess.get_inputs()}
    feed = {}
    for k, v in inputs.items():
        arr = v[0]
        if expected_rank.get(k) == 0 and arr.ndim == 1:
            arr = np.asarray(arr.reshape(()), dtype=arr.dtype)  # (1,) -> rank-0
        feed[k] = arr
    return sess.run(None, feed)[0]


def pick_device(client, substring: str):
    devices = client.get_devices()
    matches = [d for d in devices if substring.lower() in d.name.lower()]
    if not matches:
        names = sorted({d.name for d in devices})
        raise SystemExit(
            f"No device matching {substring!r}. Available:\n  " + "\n  ".join(names)
        )
    return matches[-1]  # newest OS variant of the matched device


def main(argv=None):
    # qai-hub's progress spinner prints emoji; Windows consoles default to
    # cp1252 and crash mid-wait without this.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=None,
                        help="Override ONNX path (default: prefill graph, or the "
                             "dynamic graph with --multi-graph)")
    parser.add_argument("--multi-graph", action="store_true",
                        help="Link prefill[1,16,D] + decode[1,1,D] as two graphs "
                             "of one weight-shared context binary")
    parser.add_argument("--hidden-size", type=int, default=1024)
    parser.add_argument("--device", default="Snapdragon X Elite",
                        help="Substring matched against the LIVE device list")
    parser.add_argument("--list-devices", action="store_true")
    parser.add_argument("--submit", action="store_true",
                        help="Actually submit jobs (uses AI Hub quota)")
    parser.add_argument("--infer", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--tol", type=float, default=1e-2,
                        help="Max abs diff allowed vs local FP32 ORT. HTP runs "
                             "fp16: expect ~1 ulp at the largest activation "
                             "(|x|~4 -> ~4e-3), so 1e-2 is the fp16-appropriate "
                             "default (measured on X Elite: max 3.879e-3)")
    args = parser.parse_args(argv)

    qai_hub = _require_client()
    D = args.hidden_size

    if args.multi_graph:
        model_path = Path(args.model or "artifacts/steering_injection_dynamic.onnx")
        models = [str(model_path), str(model_path)]
        graph_names = ["prefill", "decode"]
        input_specs = [
            {"hidden": ((1, 16, D), "float32"), "steering": ((1, 1, D), "float32"),
             "alpha": ((1,), "float32"), "mask": ((1, 16, 1), "float32")},
            {"hidden": ((1, 1, D), "float32"), "steering": ((1, 1, D), "float32"),
             "alpha": ((1,), "float32"), "mask": ((1, 1, 1), "float32")},
        ]
        infer_seq = 16
    else:
        model_path = Path(args.model or "artifacts/steering_injection_prefill.onnx")
        models = str(model_path)
        graph_names = None
        input_specs = None
        infer_seq = 16

    if not model_path.exists():
        raise SystemExit(f"{model_path} not found — run steering_poc.export_onnx first")

    if args.list_devices or args.submit or True:
        # Client() reads ~/.qai_hub/client.ini; may fail without a token.
        try:
            client = qai_hub.Client()
            devices_ok = True
        except Exception as e:
            client, devices_ok = None, False
            client_err = e

    if args.list_devices:
        if not devices_ok:
            raise SystemExit(f"Cannot query devices: {client_err}")
        for d in client.get_devices():
            print(f"{d.name:<40} os={d.os:<12} attrs={','.join(d.attributes)}")
        return

    device = None
    if devices_ok:
        try:
            device = pick_device(client, args.device)
            print(f"Device: {device.name} (os={device.os})")
        except SystemExit:
            raise
        except Exception as e:
            if args.submit:
                raise
            print(f"Device: live lookup failed ({type(e).__name__})")
    elif args.submit:
        raise SystemExit(
            f"qai-hub client not configured: {client_err}\n"
            "Run: qai-hub configure --api_token <token>"
        )

    print(f"Model:  {model_path}"
          + (f" as graphs {graph_names}" if graph_names else ""))

    if not args.submit:
        target = device.name if device else f"<live match for {args.device!r}>"
        print("\nDRY RUN — nothing submitted. Would do:")
        print(f"  compile+link: qai_hub.submit_compile_and_link_jobs(")
        print(f"      models={models!r},")
        if graph_names:
            print(f"      graph_names={graph_names}, input_specs=<prefill/decode>,")
        print(f"      device={target!r})   # compiles as qnn_dlc, links to a "
              "QNN context binary")
        if args.infer:
            print("  inference: qai_hub.submit_inference_job(link_job.get_target_model(), ...)")
        if args.profile:
            print("  profile:   qai_hub.submit_profile_job(link_job.get_target_model(), ...)")
        print("\nRe-run with --submit to execute.")
        return

    record: dict = {"qai_hub_version": str(qai_hub.__version__),
                    "device": device.name, "model": str(model_path),
                    "multi_graph": bool(args.multi_graph), "jobs": {}}

    print("Submitting compile+link jobs...")
    result = client.submit_compile_and_link_jobs(
        models=models,
        device=device,
        name="steering-injection",
        input_specs=input_specs,
        graph_names=graph_names,
    )
    compile_jobs, link_job = result[0], result[1]
    record["jobs"]["compile"] = [
        {"id": j.job_id, "url": j.url} for j in compile_jobs
    ]
    for j in compile_jobs:
        print(f"  compile job: {j.url}")
    if link_job is None:
        save_json(record, "artifacts/ai_hub_jobs.json")
        raise SystemExit("No link job returned — see compile job URLs above.")
    record["jobs"]["link"] = {"id": link_job.job_id, "url": link_job.url}
    print(f"  link job:    {link_job.url}")

    linked = link_job.get_target_model()  # blocks until link completes
    if linked is None:
        save_json(record, "artifacts/ai_hub_jobs.json")
        raise SystemExit(f"Link FAILED — see {link_job.url}")
    print("  compile+link OK")

    if args.infer:
        inputs = _reference_inputs(model_path, infer_seq)
        print("Submitting inference job on linked model...")
        inf_job = client.submit_inference_job(
            model=linked, device=device, inputs=inputs
        )
        record["jobs"]["inference"] = {"id": inf_job.job_id, "url": inf_job.url}
        print(f"  inference job: {inf_job.url}")
        device_out = inf_job.download_output_data()
        dev = np.asarray(list(device_out.values())[0][0])
        ref = _local_ort_reference(model_path, inputs)
        max_err = float(np.abs(dev - ref).max())
        mean_err = float(np.abs(dev - ref).mean())
        record["inference_check"] = {
            "max_abs_err_vs_local_ort": max_err,
            "mean_abs_err_vs_local_ort": mean_err,
            "tolerance": args.tol,
            "passed": max_err <= args.tol,
        }
        print(f"  device vs local ORT: max={max_err:.3e} mean={mean_err:.3e} "
              f"({'PASS' if max_err <= args.tol else 'FAIL'} at tol={args.tol})")

    if args.profile:
        print("Submitting profile job on linked model...")
        prof_job = client.submit_profile_job(model=linked, device=device)
        record["jobs"]["profile"] = {"id": prof_job.job_id, "url": prof_job.url}
        print(f"  profile job: {prof_job.url}")

    save_json(record, "artifacts/ai_hub_jobs.json")
    print("\nSaved job record -> artifacts/ai_hub_jobs.json")


if __name__ == "__main__":
    main()
