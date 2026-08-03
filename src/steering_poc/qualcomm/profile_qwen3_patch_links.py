"""Profile already-linked Qwen3-1.7B patch context binaries on AI Hub.

This reads the committed export summary, recovers each LinkJob by ID, fetches
its target model, and optionally submits a ProfileJob for each context binary.

Nothing is submitted unless --submit is passed.

Usage:
    python -m steering_poc.qualcomm.profile_qwen3_patch_links --phases alpha_4
    python -m steering_poc.qualcomm.profile_qwen3_patch_links --phases alpha_4 --submit
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ..common import save_json
from .submit_ai_hub import _require_client


def _ops_by_unit(profile: dict) -> dict[str, int]:
    units: dict[str, int] = {}
    for op in profile.get("execution_detail", []):
        unit = op.get("compute_unit", "?")
        units[unit] = units.get(unit, 0) + 1
    return units


def _graph_name(summary: dict, graph_type: str, part: int, num_parts: int) -> str:
    context = str(summary["context_lengths"]).split(",")[0].strip()
    seqs = [s.strip() for s in str(summary["sequence_lengths"]).split(",")]
    if graph_type == "prompt":
        seq = next((s for s in seqs if s != "1"), seqs[-1])
    elif graph_type == "token":
        seq = "1"
    else:
        raise ValueError(f"Unsupported graph type: {graph_type}")
    return f"{graph_type}_ar{seq}_cl{context}_{part}_of_{num_parts}"


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary",
        default="docs/results/qwen3_1_7b_patch/export_summary.json",
        help="Export summary containing phase link job IDs",
    )
    parser.add_argument(
        "--phases",
        nargs="+",
        default=["alpha_4"],
        choices=["baseline", "alpha_0", "alpha_4"],
        help="Phases to profile",
    )
    parser.add_argument(
        "--graphs",
        nargs="+",
        default=["prompt"],
        choices=["prompt", "token"],
        help="Graph types to profile from each multi-graph context binary",
    )
    parser.add_argument(
        "--out",
        default="artifacts/qwen3_1_7b_patch/profile_results.json",
        help="Where to write profile job results",
    )
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args(argv)

    summary_path = Path(args.summary)
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    qai_hub = _require_client()
    client = qai_hub.Client()

    record: dict = {
        "source_summary": str(summary_path),
        "qai_hub_version": str(qai_hub.__version__),
        "phases": {},
    }

    for phase in args.phases:
        phase_data = data["phases"][phase]
        link_ids = phase_data["link_jobs"]
        print(f"Phase {phase}: {len(link_ids)} linked context binaries")
        record["phases"][phase] = {"links": [], "profiles": []}

        for idx, link_id in enumerate(link_ids, start=1):
            link_job = client.get_job(link_id, qai_hub.JobType.LINK)
            status = link_job.get_status()
            target = link_job.get_target_model()
            link_record = {
                "part": idx,
                "link_job_id": link_id,
                "link_job_url": link_job.url,
                "link_status": getattr(status, "code", str(status)),
                "target_model_id": getattr(target, "model_id", None),
                "target_model_name": getattr(target, "name", None),
                "device": getattr(getattr(link_job, "device", None), "name", None),
            }
            record["phases"][phase]["links"].append(link_record)
            print(
                f"  part{idx}: link={link_id} target={link_record['target_model_id']} "
                f"device={link_record['device']}"
            )

            for graph_type in args.graphs:
                graph_name = _graph_name(data, graph_type, idx, len(link_ids))
                options = f"--qnn_option context_enable_graphs={graph_name}"
                if not args.submit:
                    print(f"    would profile graph={graph_name} options={options!r}")
                    continue

                if target is None:
                    raise SystemExit(f"{phase} part{idx}: no target model from {link_id}")
                device = link_job.device
                profile_job = client.submit_profile_job(
                    model=target,
                    device=device,
                    name=f"qwen3-1.7b-steering-{phase}-part{idx}-{graph_type}",
                    options=options,
                )
                print(f"    profile job ({graph_name}): {profile_job.url}")
                profile = profile_job.download_profile()
                status = profile_job.get_status()
                summary = profile.get("execution_summary", {}) if profile else {}
                profile_record = {
                    "part": idx,
                    "graph_type": graph_type,
                    "graph_name": graph_name,
                    "options": options,
                    "profile_job_id": profile_job.job_id,
                    "profile_job_url": profile_job.url,
                    "profile_status": getattr(status, "code", str(status)),
                    "profile_message": getattr(status, "message", ""),
                    "estimated_inference_time_us": summary.get("estimated_inference_time"),
                    "peak_memory_bytes": summary.get("estimated_inference_peak_memory"),
                    "ops_by_compute_unit": _ops_by_unit(profile or {}),
                }
                units = profile_record["ops_by_compute_unit"]
                profile_record["all_ops_on_npu"] = bool(units and set(units) == {"NPU"})
                record["phases"][phase]["profiles"].append(profile_record)
                save_json(record, args.out)

    if not args.submit:
        print("\nDRY RUN - no profile jobs submitted. Re-run with --submit to profile.")

    save_json(record, args.out)
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
