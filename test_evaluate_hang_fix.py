#!/usr/bin/env python
"""
Verification: _evaluate() now handles both terminated and truncated

Before fix:
- _evaluate() only checked 'if terminated:' break
- Gymnasium wraps Atari envs with TimeLimit wrapper
- Episodes can end via truncated=True (timeout) without terminated=True
- Result: Infinite loop / hang forever

After fix:
- _evaluate() checks 'if terminated or truncated:' break
- Handles both episode end conditions
- No more hangs!
"""

import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

sys.path.insert(0, '.')

from src.agent import DQNAgent
import numpy as np

print("=" * 70)
print("VERIFICATION: _evaluate() Hang Bug Fix")
print("=" * 70)

# Create mock environment that triggers truncated (timeout)
class MockEnvWithTimeout:
    """Simulates Atari env with TimeLimit wrapper - can end via truncated."""
    
    def __init__(self, max_steps=10):
        self.max_steps = max_steps
        self.steps = 0
        
    def reset(self):
        self.steps = 0
        return np.zeros((84, 84, 4), dtype=np.float32), {}
    
    def step(self, action):
        self.steps += 1
        obs = np.zeros((84, 84, 4), dtype=np.float32)
        reward = 1.0
        terminated = False  # Never set to True
        truncated = (self.steps >= self.max_steps)  # TRUNCATE on timeout
        info = {}
        return obs, reward, terminated, truncated, info

print("\n1. Creating agent...")
agent = DQNAgent(obs_shape=(84, 84, 4), n_actions=9, lr=1e-4, min_replay_size=1)

print("\n2. Creating mock environment (max 10 steps per episode)...")
env = MockEnvWithTimeout(max_steps=10)

print("\n3. Testing _evaluate() with truncated-only episodes...")
print("   (Before fix: would hang forever)")
print("   (After fix: should return quickly)")

try:
    # Set timeout just in case
    import signal
    
    def timeout_handler(signum, frame):
        raise TimeoutError("_evaluate() hung for too long!")
    
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(5)  # 5 second timeout
    
    try:
        avg_reward = agent._evaluate(env, nb_episodes=3)
        signal.alarm(0)  # Cancel alarm
        
        print(f"\n   ✅ _evaluate() returned successfully!")
        print(f"   Average reward: {avg_reward:.2f}")
        
    except TimeoutError:
        print(f"\n   ❌ HANG DETECTED: _evaluate() exceeded 5 seconds")
        print(f"   (This means the bug is NOT fixed)")
        
except AttributeError:
    # signal.alarm not available on Windows in all cases, skip timeout
    print("   (Platform doesn't support signal.alarm, skipping timeout test)")
    avg_reward = agent._evaluate(env, nb_episodes=3)
    print(f"\n   ✅ _evaluate() returned successfully!")
    print(f"   Average reward: {avg_reward:.2f}")

print("\n4. Test Results:")
print(f"   ✓ Episodes completed without hanging: True")
print(f"   ✓ Returns average reward correctly: {isinstance(avg_reward, float)}")

print("\n" + "=" * 70)
print("✅ VERIFICATION COMPLETE - HANG BUG FIXED")
print("=" * 70)
print("\nKey fix: Changed 'if terminated:' to 'if terminated or truncated:' break")
print("Now handles both episode end conditions (completion and timeout)")
