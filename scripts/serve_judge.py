#!/usr/bin/env python
"""Start the local judge server: vLLM serving the panel's local judge with the
exact candidate-distribution logits processor.

    uv run python scripts/serve_judge.py --gpus 6,7            # Llama-3.1-70B, TP=2
    uv run python scripts/serve_judge.py --model meta-llama/Llama-3.1-8B-Instruct --gpus 7

Then evaluate with `judges=panel_vllm 'plugins.active=[local_judge,vllm_backend]'`.
The server is local (http://127.0.0.1:<port>/v1); nothing is sent off the box.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile

from plugins.local_judge.measure.vllm_processor import ENV_VAR, write_candidate_spec


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="meta-llama/Llama-3.1-70B-Instruct")
    ap.add_argument("--gpus", default=None, help="CUDA_VISIBLE_DEVICES for the server, e.g. 6,7")
    ap.add_argument("--tensor-parallel-size", type=int, default=None,
                    help="default: number of --gpus")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    ap.add_argument("--max-model-len", type=int, default=4096,
                    help="judge prompts are a rubric + one answer; 4096 is ample")
    ap.add_argument("--max-num-seqs", type=int, default=512)
    ap.add_argument("--dry-run", action="store_true")
    args, extra = ap.parse_known_args()

    # Unique per launch: a name derived from the model alone is shared by every
    # user on the box, so another user's file in /tmp blocks the start
    # (PermissionError) -- or, if writable, gets overwritten under their server.
    fd, spec_path = tempfile.mkstemp(
        prefix=f"judge_candidates_{args.model.replace('/', '_')}_", suffix=".json")
    os.close(fd)
    spec = write_candidate_spec(args.model, spec_path)
    print(f"[serve_judge] {args.model}: {len(spec['candidates'])} candidate tokens, "
          f"sink={spec['sink']}, spec -> {spec_path}")

    env = dict(os.environ, **{ENV_VAR: spec_path})
    if args.gpus is not None:
        env["CUDA_VISIBLE_DEVICES"] = args.gpus
    tp = args.tensor_parallel_size or (len(args.gpus.split(",")) if args.gpus else 1)
    cmd = [
        "vllm", "serve", args.model,
        "--port", str(args.port),
        "--tensor-parallel-size", str(tp),
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        "--max-model-len", str(args.max_model_len),
        "--max-num-seqs", str(args.max_num_seqs),
        "--max-logprobs", str(spec["n_logprobs"]),
        "--logprobs-mode", "processed_logprobs",
        "--logits-processors",
        "plugins.local_judge.measure.vllm_processor:CandidateLogitsProcessor",
        "--enable-prefix-caching",
        "--served-model-name", args.model,
        *extra,
    ]
    print("[serve_judge]", " ".join(cmd), flush=True)
    if args.dry_run:
        return
    sys.exit(subprocess.call(cmd, env=env))


if __name__ == "__main__":
    main()
