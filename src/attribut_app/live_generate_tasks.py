import streamlit as st
import pandas as pd
import torch
import random
from datetime import datetime, timedelta


def generate_random_tasks(num_tasks=5):
    """Generate synthetic tasks with realistic attributes"""
    tasks = []
    priority_weights = {"High": 1.0, "Medium": 0.5, "Low": 0.2}

    for i in range(num_tasks):
        # Generate random deadline within next 7 days
        deadline = datetime.now() + timedelta(hours=random.randint(1, 168))

        tasks.append({
            "Task ID": f"TASK-{i+1:03d}",
            "Priority": random.choice(list(priority_weights.keys())),
            "Duration (hrs)": round(random.uniform(0.15, 3.0), 2),
            "Deadline": deadline.strftime("%Y-%m-%d %H:%M"),
            "Dependency": random.choice([True, False]),
            "Energy Level": random.choice(["Low", "Medium", "High"])
        })
    return tasks

def create_task_tensor(tasks, current_time):
    """Convert tasks to normalized PyTorch tensor"""
    # Normalize current time to [0, 1] within working hours (9:00-16:30)
    normalized_time = (current_time.hour - 9) + (current_time.minute/60)/7.5

    tensor_data = [normalized_time]

    for task in tasks:
        # Priority mapping
        priority_map = {"High": 1.0, "Medium": 0.5, "Low": 0.2}

        # Deadline urgency (hours remaining / 168)
        deadline = datetime.strptime(task["Deadline"], "%Y-%m-%d %H:%M")
        hours_remaining = (deadline - current_time).total_seconds() / 3600
        deadline_urgency = max(0, min(1, hours_remaining / 168))

        # Duration normalization (max 8 hours)
        duration_norm = task["Duration (hrs)"] / 8.0

        # Feature vector for each task
        task_features = [
            priority_map[task["Priority"]],
            deadline_urgency,
            duration_norm,
            1.0 if task["Dependency"] else 0.0
        ]

        tensor_data.extend(task_features)

    # Add workday preferences (Mon-Fri) and working hours
    tensor_data.extend([1,1,1,1,1,0,0])  # Workdays mask
    tensor_data.extend([9.0, 16.5])       # Preferred hours

    return torch.tensor([tensor_data], dtype=torch.float32)

# Streamlit interface
st.title("AI Task Scheduler Simulator")

# Generate random tasks
num_tasks = st.slider("Number of tasks", 3, 10, 5)
tasks = generate_random_tasks(num_tasks)

# Display as DataFrame
df = pd.DataFrame(tasks).set_index("Task ID")
st.subheader("Generated Tasks")
st.dataframe(df.style.highlight_max(axis=0, color="#f1f5f9"), use_container_width=True)

# Create tensor with current time
current_time = datetime.now()
tensor = create_task_tensor(tasks, current_time)

# Display tensor components
st.subheader("Tensor Representation")
st.markdown("**Tensor Structure:** `[Current Time] + [Task Features]*N + [Workdays] + [Hours]`")
st.write(f"Shape: {tensor.shape}")
st.code(f"{tensor.detach().numpy()}", language="python")

# Add download button
st.download_button(
    label="Download Tensor",
    data=str(tensor.detach().numpy()),
    file_name="task_tensor.pth",
    mime="application/octet-stream"
)
