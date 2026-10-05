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
