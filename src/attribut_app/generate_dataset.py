import pandas as pd
from faker import Faker
import random
from datetime import datetime, timedelta
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import multiprocessing

fake = Faker()

# User personas
PERSONAS = {
    "morning_person": {"productive_hours": (6, 12)},
    "night_owl": {"productive_hours": (18, 23)},
    "balanced": {"productive_hours": (9, 17)},
}

def generate_tasks(num_tasks=10):
    tasks = []
    for _ in range(num_tasks):
        task = {
            "id": fake.uuid4(),
            "name": fake.bs(),  # e.g., "implement neural net"
            "deadline": fake.date_time_between(start_date="+1d", end_date="+7d"),
            "priority": random.choice([1, 2, 3]),  # 1 = highest
            "duration": random.randint(30, 240),  # minutes
            "dependencies": []  # Optional: simulate task dependencies
        }
        tasks.append(task)
    return tasks

def generate_calendar_events(persona, start_date, days=7):
    productive_start, productive_end = persona["productive_hours"]
    events = []
    for _ in range(random.randint(3, 7)):  # 3-7 events per day
        start = fake.date_time_between(
            start_date=start_date.replace(hour=productive_start, minute=0),
            end_date=start_date.replace(hour=productive_end, minute=0)
        )
        duration = random.choice([30, 60, 90])
        events.append({
            "start": start,
            "end": start + timedelta(minutes=duration),
            "type": random.choice(["meeting", "break", "focus_block"])
        })
    return events

def calculate_reward(state, action, current_time=None):
    """
    Calculates reward for scheduling a task at a specific time.

    Args:
        state (dict): Current environment state containing:
            - tasks (list): Unscheduled tasks (with any scheduled ones removed)
            - calendar (list): Existing scheduled items
            - preferences (dict): User preferences, e.g. productive_hours
        action (dict): Scheduling action containing:
            - task_id: ID of task being scheduled
            - time_block: (start_time, end_time) tuple
        current_time (datetime): Current simulation time (unused)

    Returns:
        float: Reward value between -1 and 1
    """
    # Find the task being scheduled
    task = next(t for t in state['tasks'] if t['id'] == action['task_id'])
    start_time, end_time = action['time_block']
    duration = (end_time - start_time).total_seconds() / 60.0  # in minutes

    # Initialize reward components
    components = {
        'priority': 0.0,
        'deadline': 0.0,
        'time_preference': 0.0,
        'conflict': 0.0,
        'duration_match': 0.0,
        'dependencies': 0.0
    }

    # 1. Task Priority (higher = better)
    components['priority'] = (4 - task['priority']) / 3.0  # Normalize 1-3 to 1.0-0.33

    # 2. Deadline Urgency
    time_to_deadline = (task['deadline'] - end_time).total_seconds() / 3600.0
    if time_to_deadline < 0:
        components['deadline'] = -1.0  # Penalize for missing deadline
    else:
        # Sigmoid curve: max reward when scheduled midway through available time
        components['deadline'] = 2 / (1 + np.exp(0.5 * time_to_deadline)) - 1

    # 3. Time Preference Alignment
    preferred_start, preferred_end = state['preferences']['productive_hours']
    start_hour = start_time.hour + start_time.minute / 60.0
    end_hour = end_time.hour + end_time.minute / 60.0
    # Calculate overlap with preferred hours
    overlap = max(0, min(end_hour, preferred_end) - max(start_hour, preferred_start))
    if end_hour - start_hour > 0:
        components['time_preference'] = overlap / (end_hour - start_hour)
    else:
        components['time_preference'] = 0

    # 4. Time Conflict Penalty
    conflict_penalty = 0
    for event in state['calendar']:
        if (start_time < event['end']) and (end_time > event['start']):
            conflict_penalty += 1
    components['conflict'] = -0.2 * conflict_penalty

    # 5. Duration Matching
    duration_diff = abs(duration - task['duration'])
    components['duration_match'] = 1 - (duration_diff / task['duration'])

    # 6. Dependency Handling
    if task.get('dependencies'):
        for dep_id in task['dependencies']:
            scheduled_time = None
            # Try to find dependency in unscheduled tasks (it might have a scheduled_time)
            dep_task = next((t for t in state['tasks'] if t['id'] == dep_id), None)
            if dep_task:
                scheduled_time = dep_task.get('scheduled_time')
            if not scheduled_time:
                # Check calendar for scheduled dependency
                scheduled_dep = next((event for event in state['calendar'] if event.get('task_id') == dep_id), None)
                if scheduled_dep:
                    scheduled_time = (scheduled_dep['start'], scheduled_dep['end'])
            if scheduled_time and scheduled_time[1] > start_time:
                components['dependencies'] -= 0.5  # Dependency violation

    # 7. User Custom Rules
    # Example: No meetings after 4 PM
    if task.get('type') == 'meeting' and start_time.hour >= 16:
        components['time_preference'] -= 0.5

    # Combine components with weights
    weights = {
        'priority': 0.3,
        'deadline': 0.25,
        'time_preference': 0.2,
        'conflict': 0.15,
        'duration_match': 0.05,
        'dependencies': 0.05
    }

    total_reward = sum(components[comp] * weights[comp] for comp in components)

    # Clip reward to [-1, 1]
    return max(-1.0, min(1.0, total_reward))

def random_time_within(preferred_start, preferred_end, date):
    """Generate random time within preferred hours on the given date."""
    start_hour = random.randint(preferred_start, preferred_end - 1)
    return datetime.combine(date.date(), datetime.min.time()).replace(
        hour=start_hour,
        minute=random.choice([0, 15, 30, 45])
    )

def simulate_transition(state, action):
    """Update state after taking an action (scheduling a task)."""
    # Find the task being scheduled
    task = next(t for t in state['tasks'] if t['id'] == action['task_id'])
    start, end = action['time_block']
    # Attach scheduled_time to task before removal (to help with dependency checks later)
    task['scheduled_time'] = (start, end)

    next_state = {
        'tasks': [t for t in state['tasks'] if t['id'] != action['task_id']],
        'calendar': state['calendar'][:],  # copy the current calendar
        'preferences': state['preferences']
    }

    # Add the scheduled task to the calendar
    next_state['calendar'].append({
        'start': start,
        'end': end,
        'type': 'scheduled_task',
        'task_id': action['task_id'],
        'name': task['name']
    })

    return next_state

def generate_synthetic_dataset(num_samples=1000):
    """Generate full synthetic RL dataset with states, actions, rewards, and next states."""
    dataset = []

    for _ in range(num_samples):
        # Generate a user persona with preferred productive hours and workdays
        persona = {
            'productive_hours': sorted([random.randint(6, 12), random.randint(13, 22)]),
            'work_days': [0, 1, 2, 3, 4]  # Monday-Friday
        }

        # Generate tasks with possible dependencies
        num_tasks = random.randint(3, 8)
        tasks = []
        for i in range(num_tasks):
            task = {
                'id': i,
                'name': fake.bs(),
                'deadline': fake.date_time_between(start_date="+1d", end_date="+7d"),
                'priority': random.choices([1, 2, 3], weights=[0.2, 0.3, 0.5])[0],
                'duration': random.choice([30, 60, 90, 120]),
                'dependencies': random.sample(range(i), k=random.randint(0, min(2, i))) if i > 0 else []
            }
            # Optionally assign a type to some tasks (to trigger custom rules)
            if random.random() < 0.3:
                task['type'] = random.choice(['meeting', 'task', 'call'])
            tasks.append(task)

        # Generate existing calendar events
        calendar = []
        for _ in range(random.randint(2, 5)):
            dt_now = datetime.now()
            start = random_time_within(persona['productive_hours'][0], persona['productive_hours'][1], dt_now)
            calendar.append({
                'start': start,
                'end': start + timedelta(minutes=random.choice([30, 60])),
                'type': random.choice(['meeting', 'break', 'call'])
            })

        state = {
            'tasks': tasks,
            'calendar': calendar,
            'preferences': persona
        }

        # Generate a random valid action (schedule a random task)
        if tasks:
            task = random.choice(tasks)
            try:
                dt_now = datetime.now()
                start_time = random_time_within(persona['productive_hours'][0], persona['productive_hours'][1], dt_now)
                end_time = start_time + timedelta(minutes=task['duration'])

                action = {
                    'task_id': task['id'],
                    'time_block': (start_time, end_time)
                }

                # Calculate reward and get the next state
                reward = calculate_reward(state, action)
                next_state = simulate_transition(state, action)

                dataset.append({
                    'state': state,
                    'action': action,
                    'reward': reward,
                    'next_state': next_state
                })
            except Exception as e:
                print(f"Skipping invalid action: {e}")

    return pd.DataFrame(dataset)

def generate_in_parallel(num_samples=1000):
    """Generate the synthetic dataset in parallel using all available CPU cores."""
    cores = multiprocessing.cpu_count()
    samples_per_core = num_samples // cores

    with ThreadPoolExecutor(max_workers=cores) as executor:
        futures = [executor.submit(generate_synthetic_dataset, samples_per_core)
                   for _ in range(cores)]
        dfs = [f.result() for f in futures]

    return pd.concat(dfs).reset_index(drop=True)

# Example usage:
# df = generate_in_parallel(1000)
# print(df.head())
if __name__ == "__main__":
    df = generate_in_parallel(1000000)
    print(df.shape)
    df.to_csv("synthetic_dataset2.csv", index=False)
