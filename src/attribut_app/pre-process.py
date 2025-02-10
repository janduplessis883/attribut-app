import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import datetime as dt
from sklearn.model_selection import train_test_split

# Helper function for safely converting string representations to dictionaries.
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

# --- Preprocessing Pipeline ---

# Assuming your DataFrame is named df and contains the columns:
# 'state', 'action', 'reward', and 'next_state'.
# Convert the string representations to dictionaries using safe_eval.
df['state'] = df['state'].apply(safe_eval)
df['action'] = df['action'].apply(safe_eval)
df['next_state'] = df['next_state'].apply(safe_eval)

# Now compute features for the state and a discrete label for the action.
df['features'] = df['state'].apply(state_to_features)
df['action_label'] = df['action'].apply(action_to_label)

# Quick check of the features and action labels.
print(df[['features', 'action_label']].head())

# Optionally, if you need them as numerical arrays:
X = np.stack(df['features'].values)      # Feature matrix
y = df['action_label'].values            # Action labels
rewards = df['reward'].values             # Rewards

# Split into training and validation sets.
train_data, val_data = train_test_split(df[['features', 'reward', 'next_state']],
                                          test_size=0.2,
                                          random_state=42)

# Debugging Tips:
print("State column types:")
print(df['state'].apply(type).value_counts())

print("First few states:")
print(df['state'].head())
