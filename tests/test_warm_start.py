import numpy as np
import torch
from stable_baselines3 import PPO
from src.env.vec_multi import TwoCarVecEnv
from src.training.train_multi import _warm_start

def test_warm_start_identity():
    env = TwoCarVecEnv()
    model = PPO("MlpPolicy", env, n_steps=128, device="cpu")

    checkpoint_path = "checkpoints/single/slipstream-single-v1-ae7d617/final"
    single = PPO.load(checkpoint_path, device="cpu")

    _warm_start(model, checkpoint_path)

    print("--- PARAMETER CHECK ---")
    for (name, multi_param), (single_name, single_param) in zip(model.policy.named_parameters(), single.policy.named_parameters()):
        assert name == single_name
        status = "untouched (fresh init)"
        if torch.equal(multi_param, single_param):
            status = "copied perfectly"
        elif "policy_net.0.weight" in name or "value_net.0.weight" in name:
            if torch.equal(multi_param[:, :11], single_param) and torch.all(multi_param[:, 11:] == 0):
                status = "zero-padded"
        print(f"{name} ({multi_param.shape}): {status}")

    print("\n--- IDENTITY TEST ---")
    # Use 19-dim obs with last 8 dims zeroed (dims 11-14 zero from original pad, 15-18 new zeros)
    obs_11 = np.random.randn(1000, 11).astype(np.float32)
    obs_multi = np.concatenate([obs_11, np.zeros((1000, 8), dtype=np.float32)], axis=1)
    obs_single = obs_11

    actions_multi, _ = model.predict(obs_multi, deterministic=True)
    actions_single, _ = single.predict(obs_single, deterministic=True)

    with torch.no_grad():
        values_multi = model.policy.predict_values(torch.as_tensor(obs_multi)).numpy()
        values_single = single.policy.predict_values(torch.as_tensor(obs_single)).numpy()

    action_diff = np.max(np.abs(actions_multi - actions_single))
    value_diff = np.max(np.abs(values_multi - values_single))

    print(f"Max action diff: {action_diff}")
    print(f"Max value diff: {value_diff}")

    np.testing.assert_allclose(actions_multi, actions_single, atol=1e-6)
    np.testing.assert_allclose(values_multi, values_single, atol=1e-6)


def test_warm_start_phase3_15dim_to_19dim_equivalence():
    checkpoint_path = "checkpoints/multi-A/slipstream-multi-A-84a9ee1/final.zip"
    phase3_model = PPO.load(checkpoint_path, device="cpu")

    env19 = TwoCarVecEnv()
    model19 = PPO("MlpPolicy", env19, device="cpu")
    _warm_start(model19, checkpoint_path)

    np.random.seed(42)
    obs_15 = np.random.randn(100, 15).astype(np.float32)
    obs_19 = np.concatenate([obs_15, np.random.randn(100, 4).astype(np.float32)], axis=1)

    act_15, _ = phase3_model.predict(obs_15, deterministic=True)
    act_19, _ = model19.predict(obs_19, deterministic=True)

    np.testing.assert_allclose(act_19, act_15, atol=1e-6)


def test_warm_start_phase4_1b_19dim_draft_weights_zero():
    checkpoint_path = "checkpoints/phase4/slipstream-p4-1b-7b1372f/final.zip"
    env19 = TwoCarVecEnv()
    model19 = PPO("MlpPolicy", env19, device="cpu")
    _warm_start(model19, checkpoint_path)

    pi_layer = getattr(model19.policy.mlp_extractor.policy_net, "0")
    vf_layer = getattr(model19.policy.mlp_extractor.value_net, "0")

    assert torch.norm(pi_layer.weight[:, 17]).item() == 0.0
    assert torch.norm(pi_layer.weight[:, 18]).item() == 0.0
    assert torch.norm(vf_layer.weight[:, 17]).item() == 0.0
    assert torch.norm(vf_layer.weight[:, 18]).item() == 0.0

