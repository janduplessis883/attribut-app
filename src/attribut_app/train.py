import numpy as np
import pandas as pd
import json
import ast
from datetime import datetime
import datetime as dt

# --- Step 1: Convert the string columns to dictionaries ---
# (Assuming your DataFrame is already loaded as df with columns 'state', 'action', 'reward', 'next_state')
def safe_eval(x):
    """Safely evaluate a string containing a dictionary with datetime objects."""
    try:
        return eval(x, {"datetime": dt, "timedelta": dt.timedelta})
    except Exception as e:
        print("Error evaluating string:", e)
        return x

# Convert the string representations back to dicts.
df['state'] = df['state'].apply(safe_eval)
df['action'] = df['action'].apply(safe_eval)
df['next_state'] = df['next_state'].apply(safe_eval)

# --- Step 2: Define your feature conversion functions ---
def state_to_features(state, max_tasks=10, time_slots=48):
    """
    Convert the state dictionary into a fixed-length numerical feature vector.
    Features include:
      - 2 features: sine & cosine encoding of current time-of-day.
      - 1 feature: normalized productive hours remaining.
      - max_tasks*6 features: for each task (priority, days until deadline, duration, dependencies,
                                  meeting flag, scheduled flag).
      - time_slots features: occupancy indicator for each 30-minute slot in a day.
    """
    now = datetime.now()
    features = np.zeros(3 + max_tasks * 6 + time_slots)

    # 1. Time of day (sine/cosine)
    hour = now.hour + now.minute / 60.0
    features[0] = np.sin(2 * np.pi * hour / 24)
    features[1] = np.cos(2 * np.pi * hour / 24)

    # 2. Productive hours remaining (normalized over 24 hours)
    productive_start, productive_end = state['preferences']['productive_hours']
    features[2] = (productive_end - hour) / 24.0

    # 3. Task features
    for i, task in enumerate(state['tasks'][:max_tasks]):
        offset = 3 + i * 6
        features[offset]     = task['priority'] / 3.0
        features[offset + 1] = (task['deadline'] - now).total_seconds() / 86400.0
        features[offset + 2] = task['duration'] / 240.0  # assuming max 4 hours (240 min)
        features[offset + 3] = len(task['dependencies']) / 5.0  # assuming max of 5 dependencies
        features[offset + 4] = 1 if 'meeting' in task.get('type', '') else 0
        features[offset + 5] = 1 if task.get('scheduled_time') else 0

    # 4. Calendar time slots (30-minute slots over a day = 48 slots)
    base_index = 3 + max_tasks * 6
    for event in state['calendar']:
        start = event['start']
        end = event['end']
        start_slot = int((start.hour * 60 + start.minute) / 30)
        duration_slots = int((end - start).total_seconds() / 1800)
        for j in range(start_slot, min(start_slot + duration_slots, time_slots)):
            features[base_index + j] = 1

    return features

def action_to_label(action, time_slots=48):
    """
    Convert the action into a discrete label.
    Label = task_id * time_slots + time_slot index (derived from the action's start time).
    """
    start_time = action['time_block'][0]
    time_slot = int((start_time.hour * 60 + start_time.minute) / 30)
    return action['task_id'] * time_slots + time_slot

# Compute features and discrete action labels.
df['features'] = df['state'].apply(state_to_features)
df['action_label'] = df['action'].apply(action_to_label)

# --- Step 3: Save Offline RL Data in RLlib Format ---
# RLlib offline training expects one transition per line, with keys:
# "obs", "actions", "rewards", "dones", "next_obs"
# For simplicity, we treat each row as a one-step episode (i.e. done=True).

offline_data_filename = "offline_data.jsonl"
with open(offline_data_filename, "w") as f:
    for _, row in df.iterrows():
        obs = row['features'].tolist()
        next_obs = state_to_features(row['next_state']).tolist()
        action = int(row['action_label'])
        reward = float(row['reward'])
        # Mark each transition as terminal.
        data = {
            "obs": obs,
            "actions": action,
            "rewards": reward,
            "dones": True,
            "next_obs": next_obs
        }
        f.write(json.dumps(data) + "\n")

# --- Step 4: Setup RLlib Offline RL Training using PyTorch and CQL ---
import ray
from ray import tune
from ray.tune.registry import register_env
from ray.rllib.algorithms.cql import CQLConfig
import gym
from gym.spaces import Box, Discrete

# Define a dummy environment (required by RLlib) matching our observation/action space.
class OfflineDummyEnv(gym.Env):
    def __init__(self, config):
        super(OfflineDummyEnv, self).__init__()
        # Our observation vector length = 3 + max_tasks*6 + time_slots = 3 + 10*6 + 48 = 111
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(111,), dtype=np.float32)
        # Our discrete action space: we encoded actions as task_id * time_slots + time_slot.
        # With max_tasks=10 and time_slots=48, total actions = 10 * 48 = 480.
        self.action_space = Discrete(480)

    def reset(self):
        # Return a dummy observation.
        return np.zeros(self.observation_space.shape, dtype=np.float32)

    def step(self, action):
        # Dummy step implementation (won't be used during offline training).
        return self.reset(), 0.0, True, {}

# Register the dummy environment with RLlib.
register_env("OfflineDummyEnv", lambda config: OfflineDummyEnv(config))

# Initialize Ray.
ray.init(ignore_reinit_error=True)

# Configure the offline RL training using CQL.
config = CQLConfig().environment(
    env="OfflineDummyEnv",
    env_config={}
).offline_data(
    input_=offline_data_filename
).framework("torch").training(
    # You can adjust additional training parameters as needed.
    # For example, learning rate, training batch size, etc.
    train_batch_size=64
)

# Build the algorithm.
algo = config.build()

# --- Step 5: Train the RL Model ---
for i in range(10):
    result = algo.train()
    print(f"Iteration {i}: reward_mean = {result.get('episode_reward_mean', 'N/A')}")

# Optionally, save the trained model.
checkpoint = algo.save("offline_rl_model")
print("Checkpoint saved at", checkpoint)

# Shut down Ray.
ray.shutdown()
