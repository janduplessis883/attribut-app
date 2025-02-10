import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import json
import torch
from torch.utils.data import Dataset, DataLoader
from datetime import datetime, timedelta
import datetime as dt
from sklearn.model_selection import train_test_split

import json
import torch
from torch.utils.data import Dataset, DataLoader



import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# -------------------------------
# 1. Define the Offline Dataset (JSONL based)
# -------------------------------
class TorchOfflineDataset(Dataset):
    def __init__(self, jsonl_file):
        self.data = []
        with open(jsonl_file, "r") as f:
            for line in f:
                self.data.append(json.loads(line))
        print(f"Loaded {len(self.data)} transitions from {jsonl_file}.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        # Convert each field to a PyTorch tensor with the appropriate type.
        return {
            'obs': torch.tensor(sample['obs'], dtype=torch.float32),
            'action': torch.tensor(sample['actions'], dtype=torch.long),
            'reward': torch.tensor(sample['rewards'], dtype=torch.float32),
            'done': torch.tensor(sample['dones'], dtype=torch.bool),
            'next_obs': torch.tensor(sample['next_obs'], dtype=torch.float32)
        }



# -------------------------------
# 2. Define the Q-Network and Helper Functions
# -------------------------------
class QNetwork(nn.Module):
    def __init__(self, obs_dim, n_actions):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(obs_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.out = nn.Linear(256, n_actions)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.out(x)  # returns Q-values for each action

def soft_update(target, source, tau):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)

# -------------------------------
# 3. Hyperparameters & Setup
# -------------------------------
obs_dim = 111      # observation dimension (as defined by your environment)
n_actions = 480    # total number of discrete actions
gamma = 0.99       # discount factor
alpha = 1.0        # conservative penalty weight (tune this!)
batch_size = 64
learning_rate = 3e-4
n_epochs = 10
target_update_tau = 0.005

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}\n")

q_net = QNetwork(obs_dim, n_actions).to(device)
target_q_net = QNetwork(obs_dim, n_actions).to(device)
target_q_net.load_state_dict(q_net.state_dict())

optimizer = optim.Adam(q_net.parameters(), lr=learning_rate)

# Assume state_to_features is defined exactly as during training.
# And assume your QNetwork class is available.
def safe_eval(x):
    """
    Evaluate a string containing a dictionary with datetime objects.
    Provides the datetime module and timedelta for evaluation.
    """
    try:
        # Use the datetime module from dt so that "datetime.datetime(...)" works.
        return eval(x, {"datetime": dt, "timedelta": dt.timedelta})
    except Exception as e:
        print("Error evaluating string:", e)
        return x

def state_to_features(state, max_tasks=10, time_slots=48):
    """
    Convert the state dictionary into a numerical feature vector.

    Feature breakdown:
      - 2 features: sine & cosine of current time-of-day.
      - 1 feature: normalized productive hours remaining.
      - max_tasks * 6 features: for each task, include:
            * normalized priority,
            * days remaining until deadline,
            * normalized duration (relative to 4 hours),
            * normalized count of dependencies,
            * binary flag if type contains 'meeting',
            * binary flag if the task is already scheduled.
      - time_slots features: one-hot encoding of occupied calendar time slots (30-minute slots).
    """
    now = datetime.now()
    # Total feature length = 3 (time context) + (max_tasks*6) + time_slots
    features = np.zeros(3 + max_tasks * 6 + time_slots)

    # 1. Time of day (sine/cosine encoding)
    hour = now.hour + now.minute / 60.0
    features[0] = np.sin(2 * np.pi * hour / 24)
    features[1] = np.cos(2 * np.pi * hour / 24)

    # 2. Productive hours remaining (normalized over 24 hours)
    productive_start, productive_end = state['preferences']['productive_hours']
    features[2] = (productive_end - hour) / 24.0

    # 3. Task features (up to max_tasks)
    for i, task in enumerate(state['tasks'][:max_tasks]):
        offset = 3 + i * 6
        # Normalize priority (assuming values 1-3)
        features[offset] = task['priority'] / 3.0
        # Days until deadline (could be negative if past due)
        features[offset + 1] = (task['deadline'] - now).total_seconds() / 86400.0
        # Duration normalized to max 4 hours (240 minutes)
        features[offset + 2] = task['duration'] / 240.0
        # Number of dependencies normalized (assuming maximum of 5)
        features[offset + 3] = len(task['dependencies']) / 5.0
        # Binary flag: does the task type contain 'meeting'?
        features[offset + 4] = 1 if 'meeting' in task.get('type', '') else 0
        # Binary flag: has the task been scheduled?
        features[offset + 5] = 1 if task.get('scheduled_time') else 0

    # 4. Calendar time slots (48 slots for 30-minute intervals in a day)
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
    Label is computed as: task_id * time_slots + time_slot index.
    """
    start_time = action['time_block'][0]
    time_slot = int((start_time.hour * 60 + start_time.minute) / 30)
    return action['task_id'] * time_slots + time_slot


def select_action(real_world_state):
    """
    real_world_state: a dictionary with the same structure as the state used during training.
    """
    # Convert your state into features.
    features = state_to_features(real_world_state)  # returns a NumPy array of shape [111]
    features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)  # add batch dim

    # Forward pass to get Q-values.
    with torch.no_grad():
        q_values = q_net(features_tensor)  # shape: [1, n_actions]

    # Choose the action with highest Q-value.
    action_index = q_values.argmax(dim=1).item()  # greedy selection

    # Optional: If you have a mapping function to decode the action index back into a more interpretable action:
    # action = decode_action(action_index)

    return action_index


def decode_action(action_index, time_slots=48):
    """
    Given an encoded action index, decode it to (task_id, time_slot).
    """
    task_id = action_index // time_slots
    time_slot = action_index % time_slots
    return task_id, time_slot

def get_timeslot_start(time_slot, base_time=None):
    """
    Convert a time_slot index to an actual datetime object.
    Assumes that the day starts at base_time (default: midnight of today).
    Each slot is 30 minutes long.
    """
    if base_time is None:
        # Define midnight of the current day.
        base_time = datetime.combine(datetime.today(), datetime.min.time())
    return base_time + timedelta(minutes=time_slot * 30)


def update_task_calendar(state, action_index, time_slots=48):
    """
    Given a state dictionary and an encoded action index, update the state:
      - Decode the action to get task_id and time_slot.
      - Update the scheduled_time for that task with both start and end times,
        where slot_end is computed as slot_start plus the task's duration.
    """
    from datetime import timedelta

    # Decode the action index.
    task_id, time_slot = decode_action(action_index, time_slots)
    print(f"Decoded action {action_index}: task_id = {task_id}, time_slot = {time_slot}")

    # Determine the start time for the time slot.
    slot_start = get_timeslot_start(time_slot)

    # Update the task if it exists.
    if task_id < len(state['tasks']):
        # Get the duration from the task or default to 30 minutes.
        duration = int(state['tasks'][task_id].get('duration', 30))
        slot_end = slot_start + timedelta(minutes=duration)
        # Update the task's scheduled_time with both start and end as a tuple.
        state['tasks'][task_id]['scheduled_time'] = (slot_start, slot_end)
        print(f"Task {task_id} scheduled from {slot_start} to {slot_end}")
    else:
        print(f"Warning: Task ID {task_id} not found in the state.")

    # Note: No calendar event is appended.
    return state
