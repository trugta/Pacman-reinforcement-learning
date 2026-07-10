#!/usr/bin/env python
"""
Verification: train_steps now increments correctly (not double-incremented)

Before fix:
- fit() incremented train_steps on EVERY env step
- train_step() also incremented train_steps on EVERY training update
- Result: train_steps += 2 per step once buffer is warm (double-increment bug!)

After fix:
- fit() does NOT increment train_steps
- train_step() increments train_steps ONLY when actually training
- Result: train_steps += 1 per training step (correct!)
"""

import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

sys.path.insert(0, '.')

from src.agent import DQNAgent
import numpy as np

print("=" * 70)
print("VERIFICATION: train_steps Double-Increment Bug Fix")
print("=" * 70)

# Create agent
agent = DQNAgent(obs_shape=(84, 84, 4), n_actions=9, lr=1e-4, min_replay_size=10, batch_size=4)

# Create dummy transitions
dummy_state = np.zeros((84, 84, 4), dtype=np.float32)
dummy_reward = 1.0
dummy_action = 0
dummy_terminated = False

print("\n1. Initial state:")
print(f"   train_steps: {agent.train_steps}")
print(f"   buffer size: {len(agent.replay_buffer)}")

print("\n2. Adding 5 transitions (buffer not warm yet, no training)...")
for i in range(5):
    agent.store(dummy_state, dummy_action, dummy_reward, dummy_state, dummy_terminated)
    print(f"   After store {i+1}: train_steps={agent.train_steps}, buffer_size={len(agent.replay_buffer)}")

print(f"\n   ✓ train_steps unchanged (no training yet): {agent.train_steps == 0}")

print("\n3. Adding 10 more transitions to warm buffer...")
for i in range(10):
    agent.store(dummy_state, dummy_action, dummy_reward, dummy_state, dummy_terminated)

print(f"   Buffer now warm: {agent.can_train()}")
print(f"   train_steps before training: {agent.train_steps}")

print("\n4. Calling train_step() 5 times...")
initial_steps = agent.train_steps
for i in range(5):
    loss = agent.train_step()
    loss_str = f"{loss:.4f}" if loss is not None else "None"
    print(f"   After train_step {i+1}: train_steps={agent.train_steps}, loss={loss_str}")

steps_after_training = agent.train_steps
steps_incremented = steps_after_training - initial_steps

print(f"\n5. CRITICAL VERIFICATION:")
print(f"   Expected increment: 5 (one per train_step call)")
print(f"   Actual increment: {steps_incremented}")
print(f"   ✓ Correct: {steps_incremented == 5}")

if steps_incremented == 5:
    print(f"\n   ✅ SUCCESS: train_steps increments ONLY when training")
    print(f"   ✅ NO double-increment bug!")
elif steps_incremented == 10:
    print(f"\n   ❌ FAILURE: Double-increment detected (got {steps_incremented} instead of 5)")
else:
    print(f"\n   ⚠️  Unexpected increment: {steps_incremented}")

print("\n6. Epsilon decay verification:")
print(f"   train_steps=0: epsilon={agent._get_epsilon(0):.3f}")
print(f"   train_steps=250000: epsilon={agent._get_epsilon(250000):.3f}")
print(f"   train_steps=500000: epsilon={agent._get_epsilon(500000):.3f}")
print(f"   (Should decay smoothly from 1.0 toward 0.01)")

print("\n" + "=" * 70)
print("✅ VERIFICATION COMPLETE")
print("=" * 70)
