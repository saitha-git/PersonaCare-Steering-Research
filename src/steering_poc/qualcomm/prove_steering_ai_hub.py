"""Device proof for the standalone steering injection op on Qualcomm AI Hub.

This submits the fixed-shape ONNX graph

    steered = hidden + alpha * mask * steering

to a Snapdragon device, runs the same inputs with alpha=0 and alpha>0, and
checks that the on-device delta matches the expected steering update. The mask
is zero for the first half of the sequence and one for the second half, so a
single run proves both identity and active steering behavior.

Nothing is submitted unless --submit is passed.

Usage:
    python -m steering_poc.qualcomm.prove_steering_ai_hub
    python -m steering_poc.qualcomm.prove_steering_ai_hub --submit --profile
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from ..common import file_sha256, save_json
from .submit_ai_hub import _local_ort_reference, _require_client, pick_device


def _load_vector(path: Path | None, hidden_size: int) -> tuple[np.ndarray, str]:
    if path is None:
        rng = np.random.default_rng(1235)
        vector = rng.standard_normal(hidden_size).astype(np.float32)
        source = "deterministic random unit vector"
    else:
        import torch

        payload = torch.load(path, map_location="cpu", weights_only=False)
        vector_t = payload["vector"] if isinstance(payload, dict) else payload
        vector = vector_t.detach().cpu().float().numpy().reshape(-1)
        source = str(path)
    if vector.size != hidden_size:
        raise SystemExit(
            f"Vector has {vector.size} elements, expected hidden_size={hidden_size}"
        )
    norm = np.linalg.norm(vector)
    if norm == 0:
        raise SystemExit("Steering vector has zero norm")
    return (vector / norm).reshape(1, 1, hidden_size).astype(np.float32), source


def _inputs(hidden_size: int, seq_len: int, alpha: float, vector: np.ndarray):
    rng = np.random.default_rng(1234)
    hidden = rng.standard_normal((1, seq_len, hidden_size)).astype(np.float32)
    mask = np.zeros((1, seq_len, 1), dtype=np.float32)
    mask[:, seq_len // 2 :, :] = 1.0
    return {
        "hidden": [hidden],
        "steering": [vector],
        "alpha": [np.array([alpha], dtype=np.float32)],
        "mask": [mask],
    }


def _first_output(output_data) -> np.ndarray:
    if output_data is None:
        raise SystemExit("Inference job did not return output data")
    return np.asarray(list(output_data.values())[0][0], dtype=np.float32)


def _metrics(a0: np.ndarray, aN: np.ndarray, inputs0: dict, alpha: float):
    hidden = inputs0["hidden"][0]
    steering = inputs0["steering"][0]
    mask = inputs0["mask"][0]
    expected_delta = alpha * mask * steering
    device_delta = aN - a0
    delta_err = np.abs(device_delta - expected_delta)
    identity_err = np.abs(a0 - hidden)
    masked = mask == 0
    active = mask == 1
    flat_dev = device_delta[active.repeat(device_delta.shape[-1], axis=-1)]
    flat_exp = expected_delta[active.repeat(expected_delta.shape[-1], axis=-1)]
    denom = np.linalg.norm(flat_dev) * np.linalg.norm(flat_exp)
    cosine = float(np.dot(flat_dev.reshape(-1), flat_exp.reshape(-1)) / denom)
    active_delta_l2 = float(np.linalg.norm(device_delta[active.repeat(device_delta.shape[-1], axis=-1)]))
    masked_delta_l2 = float(np.linalg.norm(device_delta[masked.repeat(device_delta.shape[-1], axis=-1)]))
    return {
        "alpha0_identity_vs_input": {
            "max_abs_err": float(identity_err.max()),
            "mean_abs_err": float(identity_err.mean()),
            "fp16_passed": bool(identity_err.max() <= 1e-2),
        },
        "alpha_delta_vs_expected": {
            "max_abs_err": float(delta_err.max()),
            "mean_abs_err": float(delta_err.mean()),
            "cosine_active_positions": cosine,
            "fp16_passed": bool(delta_err.max() <= 1e-2),
        },
        "mask_check": {
            "masked_delta_l2": masked_delta_l2,
            "active_delta_l2": active_delta_l2,
            "active_to_masked_l2_ratio": (
                None if masked_delta_l2 == 0 else active_delta_l2 / masked_delta_l2
            ),
            "ratio_note": (
                "undefined because masked positions had exactly zero delta"
                if masked_delta_l2 == 0
                else None
            ),
        },
        "overall_passed": bool(identity_err.max() <= 1e-2 and delta_err.max() <= 1e-2),
    }


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="artifacts/steering_injection_prefill.onnx")
    parser.add_argument("--vector", default="artifacts/vector_layer_14.pt")
    parser.add_argument("--hidden-size", type=int, default=1024)
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--alpha", type=float, default=4.0)
    parser.add_argument("--device", default="Snapdragon X Elite")
    parser.add_argument("--out", default="artifacts/steering_device_proof.json")
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args(argv)

    model_path = Path(args.model)
    if not model_path.exists():
        raise SystemExit(f"{model_path} not found - run steering_poc.export_onnx first")
    vector_path = Path(args.vector) if args.vector else None
    if vector_path is not None and not vector_path.exists():
        raise SystemExit(f"{vector_path} not found - run steering_poc.extract first")

    vector, vector_source = _load_vector(vector_path, args.hidden_size)
    inputs0 = _inputs(args.hidden_size, args.seq_len, 0.0, vector)
    inputsN = _inputs(args.hidden_size, args.seq_len, args.alpha, vector)

    qai_hub = _require_client()
    client = qai_hub.Client()
    device = pick_device(client, args.device)

    print(f"Device: {device.name} (os={device.os})")
    print(f"Model:  {model_path}")
    print(f"Vector: {vector_source}")
    print(f"Mask:   first {args.seq_len // 2} tokens off, last {args.seq_len - args.seq_len // 2} tokens on")

    if not args.submit:
        print("\nDRY RUN - nothing submitted. Re-run with --submit to compile/link, run alpha=0 and alpha>0 inference jobs, and write the proof JSON.")
        return

    record: dict = {
        "qai_hub_version": str(qai_hub.__version__),
        "device": device.name,
        "device_os": device.os,
        "model": str(model_path),
        "model_sha256": file_sha256(str(model_path)),
        "vector_source": vector_source,
        "vector_sha256": file_sha256(str(vector_path)) if vector_path else None,
        "hidden_size": args.hidden_size,
        "seq_len": args.seq_len,
        "alpha": args.alpha,
        "mask": "first half 0, second half 1",
        "jobs": {},
    }

    print("Submitting compile+link jobs...")
    compile_jobs, link_job = client.submit_compile_and_link_jobs(
        models=str(model_path),
        device=device,
        name="steering-device-proof",
    )
    record["jobs"]["compile"] = [{"id": j.job_id, "url": j.url} for j in compile_jobs]
    if link_job is not None:
        record["jobs"]["link"] = {"id": link_job.job_id, "url": link_job.url}
        linked = link_job.get_target_model()
    else:
        linked = compile_jobs[0].get_target_model()
    if linked is None:
        save_json(record, args.out)
        raise SystemExit("Compile/link failed; see recorded job URLs")
    print(f"  linked model: {linked.model_id}")

    local0 = _local_ort_reference(model_path, inputs0)
    localN = _local_ort_reference(model_path, inputsN)

    print("Submitting alpha=0 inference job...")
    job0 = client.submit_inference_job(model=linked, device=device, inputs=inputs0)
    record["jobs"]["inference_alpha_0"] = {"id": job0.job_id, "url": job0.url}
    dev0 = _first_output(job0.download_output_data())

    print(f"Submitting alpha={args.alpha:g} inference job...")
    jobN = client.submit_inference_job(model=linked, device=device, inputs=inputsN)
    record["jobs"][f"inference_alpha_{args.alpha:g}"] = {
        "id": jobN.job_id,
        "url": jobN.url,
    }
    devN = _first_output(jobN.download_output_data())

    record["local_ort_delta_check"] = _metrics(local0, localN, inputs0, args.alpha)
    record["device_delta_check"] = _metrics(dev0, devN, inputs0, args.alpha)
    record["device_vs_local_ort"] = {
        "alpha_0_max_abs_err": float(np.abs(dev0 - local0).max()),
        "alpha_0_mean_abs_err": float(np.abs(dev0 - local0).mean()),
        f"alpha_{args.alpha:g}_max_abs_err": float(np.abs(devN - localN).max()),
        f"alpha_{args.alpha:g}_mean_abs_err": float(np.abs(devN - localN).mean()),
        "fp16_passed": bool(
            np.abs(dev0 - local0).max() <= 1e-2 and np.abs(devN - localN).max() <= 1e-2
        ),
    }

    if args.profile:
        print("Submitting profile job...")
        prof_job = client.submit_profile_job(model=linked, device=device)
        record["jobs"]["profile"] = {"id": prof_job.job_id, "url": prof_job.url}
        prof = prof_job.download_profile()
        summary = prof.get("execution_summary", {})
        units: dict[str, int] = {}
        for op in prof.get("execution_detail", []):
            cu = op.get("compute_unit", "?")
            units[cu] = units.get(cu, 0) + 1
        record["profile_summary"] = {
            "estimated_inference_time_us": summary.get("estimated_inference_time"),
            "peak_memory_bytes": summary.get("estimated_inference_peak_memory"),
            "ops_by_compute_unit": units,
            "all_ops_on_npu": bool(units and set(units) == {"NPU"}),
        }

    save_json(record, args.out)
    print(f"\nSaved -> {args.out}")
    print(f"Device proof overall: {'PASS' if record['device_delta_check']['overall_passed'] else 'FAIL'}")


if __name__ == "__main__":
    main()
