import json
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import torch

from src.env.multi_racing_env import MultiRacingEnv
from scripts.run_diagnostics import load_model
from scripts.evaluate_p4_1b import evaluate_model_on_block

def eval_single_checkpoint(item):
    label, path_str = item
    path = Path(path_str)
    if not path.exists():
        print(f"Missing {label} at {path_str}", flush=True)
        return label, None

    torch.set_num_threads(1)
    env = MultiRacingEnv(spawn_offset_idx=15)
    print(f"Starting {label}...", flush=True)
    model = load_model(str(path), env)

    det_prim = evaluate_model_on_block(model, env, seed_start=1000, n_episodes=50, stochastic=False)
    stoch_prim = evaluate_model_on_block(model, env, seed_start=1000, n_episodes=50, stochastic=True)
    det_conf = evaluate_model_on_block(model, env, seed_start=2000, n_episodes=50, stochastic=False)
    stoch_conf = evaluate_model_on_block(model, env, seed_start=2000, n_episodes=50, stochastic=True)

    print(f"Finished {label}!", flush=True)
    return label, {
        "primary": {
            "det": det_prim,
            "stoch": stoch_prim,
        },
        "confirmation": {
            "det": det_conf,
            "stoch": stoch_conf,
        }
    }

def main():
    ckpt_dir = Path("checkpoints/phase4/slipstream-p4-1b-7b1372f")
    if not ckpt_dir.exists():
        dirs = sorted(list(Path("checkpoints/phase4").glob("slipstream-p4-1b*")))
        if dirs:
            ckpt_dir = dirs[-1]

    ckpts = [
        ("Phase 3 Final", "checkpoints/multi-A/slipstream-multi-A-84a9ee1/final.zip"),
        ("4.1-legacy", "checkpoints/phase4/slipstream-p4-1-0fdfaac/final.zip"),
        ("4.1b 100k", str(ckpt_dir / "model_100000.zip")),
        ("4.1b 200k", str(ckpt_dir / "model_200000.zip")),
        ("4.1b 300k", str(ckpt_dir / "model_300000.zip")),
        ("4.1b 400k", str(ckpt_dir / "model_400000.zip")),
        ("4.1b 500k", str(ckpt_dir / "model_500000.zip")),
        ("4.1b Final", str(ckpt_dir / "final.zip")),
    ]

    with Pool(2) as p:
        results_list = p.map(eval_single_checkpoint, ckpts)

    results = {label: data for label, data in results_list if data is not None}
    out_file = Path("scratch/p4_1b_checkpoint_eval_results.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(results, indent=2))
    print(f"All evaluations complete! Saved to {out_file}", flush=True)

if __name__ == "__main__":
    main()
