import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from environment.reach_env import FrankReachEnv

env = FrankReachEnv(render_mode=None)
obs, info = env.reset()

print(f"Observation: {obs}")
print(f"Target: {env.target_pos}")
print(f"EE pos at home: {obs[14:17]}")
print(f"Distance at home: {np.linalg.norm(obs[14:17] - env.target_pos):.3f}m")

print("\nZero action test:")

for i in range(10):
    obs, reward, term, trunc, info = env.step(np.zeros(7))
    print(f"  step {i}: distance={info['distance']:.4f} reward={reward:.4f}")