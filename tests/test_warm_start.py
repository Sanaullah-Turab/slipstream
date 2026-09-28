import numpy as np
import torch
from stable_baselines3 import PPO
from src.env.vec_multi import TwoCarVecEnv
from src.training.train_multi import _warm_start

def test_warm_start_identity():
    # 1. Create a dummy multi-agent env and model
    env = TwoCarVecEnv()
    model = PPO("MlpPolicy", env, n_steps=128, device="cpu")
    
    # 2. Warm start
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
        
        # Value head is not copied in _warm_start! Wait, is it?
        # Let's see if we copy value head in _warm_start
        print(f"{name} ({multi_param.shape}): {status}")

    print("\n--- IDENTITY TEST ---")
    obs = np.random.randn(1000, 15).astype(np.float32)
    obs_single = obs[:, :11]
    
    actions_multi, _ = model.predict(obs, deterministic=True)
    actions_single, _ = single.predict(obs_single, deterministic=True)
    
    # Check values
    with torch.no_grad():
        values_multi = model.policy.predict_values(torch.as_tensor(obs)).numpy()
        values_single = single.policy.predict_values(torch.as_tensor(obs_single)).numpy()
        
    action_diff = np.max(np.abs(actions_multi - actions_single))
    value_diff = np.max(np.abs(values_multi - values_single))
    
    print(f"Max action diff: {action_diff}")
    print(f"Max value diff: {value_diff}")
    
    np.testing.assert_allclose(actions_multi, actions_single, atol=1e-6)
    np.testing.assert_allclose(values_multi, values_single, atol=1e-6)
