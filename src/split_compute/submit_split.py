"""Two-device pipeline verification on Qualcomm AI Hub (opt-in: --submit).

  PartA (embed + layers 0..k-1)  -> compiled & run on a PHONE device
  PartB (layers k..N-1 + head)   -> compiled & run on a LAPTOP device

The boundary hidden state produced on the phone is downloaded and fed as the
input of the laptop inference job — AI Hub has no live device-to-device
transport, so the cloud round trip stands in for the network. This verifies
functional/numerical viability of cross-device pipelining on real silicon;
it does NOT measure live transport latency, streaming decode, or power.

Usage:
    python -m split_compute.submit_split                        # dry run
    python -m split_compute.submit_split --submit \
        [--phone "Snapdragon 8 Elite QRD"] [--laptop "Snapdragon X Elite CRD"]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from steering_poc.common import save_json
from steering_poc.qualcomm.submit_ai_hub import _require_client, pick_device

from .verify_local import chain_metrics


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", default="artifacts/split")
    parser.add_argument("--phone", default="Snapdragon 8 Elite QRD")
    parser.add_argument("--laptop", default="Snapdragon X Elite CRD")
    parser.add_argument("--submit", action="store_true",
                        help="Actually submit jobs (uses AI Hub quota)")
    parser.add_argument("--profile", action="store_true",
                        help="Also profile each half on its device")
    args = parser.parse_args(argv)

    d = Path(args.dir)
    for req in ("part_a.onnx", "part_b.onnx", "input_ids.npy", "ref_logits.npy"):
        if not (d / req).exists():
            raise SystemExit(
                f"{d / req} missing — run split_compute.export_split then "
                "split_compute.verify_local first."
            )
    meta = json.loads((d / "split_meta.json").read_text())

    qai_hub = _require_client()
    client = qai_hub.Client()
    phone = pick_device(client, args.phone)
    laptop = pick_device(client, args.laptop)

    print(f"PartA (layers 0..{meta['split_layer'] - 1} + embed)  -> {phone.name}")
    print(f"PartB (layers {meta['split_layer']}..{meta['num_layers'] - 1} + head) "
          f"-> {laptop.name}")
    print(f"Boundary tensor: {meta['boundary_tensor']}")

    if not args.submit:
        print("\nDRY RUN — nothing submitted. Would do:")
        print(f"  1. compile+link part_a.onnx for {phone.name!r}")
        print(f"  2. compile+link part_b.onnx for {laptop.name!r}")
        print("  3. inference PartA on phone with the saved input_ids")
        print("  4. download boundary hidden state; inference PartB on laptop")
        print("  5. compare chained logits vs local fp32 reference")
        if args.profile:
            print("  6. profile both halves on their devices")
        print("Re-run with --submit to execute.")
        return

    record: dict = {"meta": meta, "phone": phone.name, "laptop": laptop.name,
                    "jobs": {}}

    def compile_for(path, device, label):
        print(f"Compiling {label} for {device.name} ...")
        compile_jobs, link_job = client.submit_compile_and_link_jobs(
            models=str(path), device=device, name=f"split-{label}",
        )
        record["jobs"][f"compile_{label}"] = [
            {"id": j.job_id, "url": j.url} for j in compile_jobs
        ]
        if link_job is not None:
            record["jobs"][f"link_{label}"] = {
                "id": link_job.job_id, "url": link_job.url
            }
            target = link_job.get_target_model()
        else:
            target = compile_jobs[0].get_target_model()
        if target is None:
            save_json(record, d / "ai_hub_split_jobs.json")
            raise SystemExit(f"{label} compile/link FAILED — see record")
        print(f"  {label} ready (model {target.model_id})")
        return target

    model_a = compile_for(d / "part_a.onnx", phone, "part_a")
    model_b = compile_for(d / "part_b.onnx", laptop, "part_b")

    ids = np.load(d / "input_ids.npy")
    print(f"Inference PartA on {phone.name} ...")
    job_a = client.submit_inference_job(
        model=model_a, device=phone, inputs={"input_ids": [ids]}
    )
    record["jobs"]["infer_part_a"] = {"id": job_a.job_id, "url": job_a.url}
    print(f"  {job_a.url}")
    out_a = job_a.download_output_data()
    if out_a is None:
        save_json(record, d / "ai_hub_split_jobs.json")
        raise SystemExit(f"PartA inference FAILED — {job_a.url}")
    boundary_device = np.asarray(list(out_a.values())[0][0], dtype=np.float32)
    np.save(d / "device_boundary.npy", boundary_device)

    local_boundary = np.load(d / "local_boundary.npy")
    b_err = float(np.abs(boundary_device - local_boundary).max())
    record["boundary_check"] = {
        "max_abs_err_vs_local_fp32": b_err,
        "shape": list(boundary_device.shape),
    }
    print(f"  boundary hidden state: shape {boundary_device.shape}, "
          f"max err vs local fp32 = {b_err:.3e}")

    print(f"Inference PartB on {laptop.name} (feeding the phone's activations)...")
    job_b = client.submit_inference_job(
        model=model_b, device=laptop, inputs={"hidden": [boundary_device]}
    )
    record["jobs"]["infer_part_b"] = {"id": job_b.job_id, "url": job_b.url}
    print(f"  {job_b.url}")
    out_b = job_b.download_output_data()
    if out_b is None:
        save_json(record, d / "ai_hub_split_jobs.json")
        raise SystemExit(f"PartB inference FAILED — {job_b.url}")
    chain_logits = np.asarray(list(out_b.values())[0][0], dtype=np.float32)
    np.save(d / "device_chain_logits.npy", chain_logits)

    ref_logits = np.load(d / "ref_logits.npy")
    m = chain_metrics(ref_logits, chain_logits)
    record["chained_vs_local_fp32"] = m
    print("\nPhone->laptop chain vs local single-device fp32 reference:")
    for k, v in m.items():
        print(f"  {k}: {v}")

    if args.profile:
        for label, model_t, device in (("part_a", model_a, phone),
                                       ("part_b", model_b, laptop)):
            print(f"Profiling {label} on {device.name} ...")
            pj = client.submit_profile_job(model=model_t, device=device)
            record["jobs"][f"profile_{label}"] = {"id": pj.job_id, "url": pj.url}
            prof = pj.download_profile()
            summary = prof.get("execution_summary", {})
            units: dict = {}
            for op in prof.get("execution_detail", []):
                cu = op.get("compute_unit", "?")
                units[cu] = units.get(cu, 0) + 1
            record[f"profile_{label}"] = {
                "estimated_inference_time_us":
                    summary.get("estimated_inference_time"),
                "peak_memory_bytes":
                    summary.get("estimated_inference_peak_memory"),
                "ops_by_compute_unit": units,
            }
            print(f"  {record[f'profile_{label}']}")

    save_json(record, d / "ai_hub_split_jobs.json")
    print(f"\nSaved -> {d / 'ai_hub_split_jobs.json'}")


if __name__ == "__main__":
    main()
