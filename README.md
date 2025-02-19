# Attribut AI Notion Task Scheduler

![image](images/streamlit.png)

## https://notion-scheduler.streamlit.app


This repository contains a system for intelligent task scheduling and rescheduling in Notion. It leverages a neural network to predict extra time required for tasks and reschedules them based on priority, deadlines, and availability in a Notion database.

## Features

### 🧠 Neural Network-Powered Task Rescheduling
- A multi-layer neural network predicts the extra time needed for tasks based on:
  - Number of times a task has been rescheduled.
  - Task priority.
  - Task duration.
  - Current time of scheduling.
- The model is trained dynamically with simulated targets to improve accuracy over time.

### 📅 Notion API Integration
- Fetches tasks from a Notion database with filters applied for tasks that are:
  - **Not started** or **In progress**.
  - Have a deadline within four days.
- Updates tasks in Notion with new scheduled times and durations.
- Preserves "Fixed Schedule" tasks that should not be moved.

### ⏳ Smart Scheduling Logic
- **Respects working hours:** Tasks are only scheduled between **09:00 and 16:00** (Monday to Friday).
- **Fixed schedule handling:** Tasks with a fixed schedule are not modified and other tasks avoid overlapping with them.
- **Priority-aware scheduling:**
  - Urgent tasks (deadline within 1 hour) are scheduled first.
  - Other tasks are scheduled based on priority.
- **Adaptive rescheduling:**
  - In-progress tasks are extended dynamically if needed.
  - The model improves over time by learning from rescheduled tasks.

### 📈 Streamlit Dashboard
- Provides an interactive interface to toggle the scheduler.
- Displays a progress bar showing the countdown to the next scheduling cycle.
- Uses **Streamlit Secrets** to store API keys securely.

### ⏲️ Automated Scheduling Loop
- Runs every **15 minutes** to fetch and reschedule tasks.
- Displays status updates and warnings about potential deadline conflicts.
- Uses a **progress bar countdown** to show time remaining until the next execution.

## How It Works
1. The script fetches tasks from Notion.
2. It processes them based on priority, deadline, and fixed schedule constraints.
3. The neural network predicts the likelihood of extra time required.
4. Tasks are rescheduled accordingly, avoiding conflicts with fixed schedules and working hours.
5. Updates are sent back to Notion.
6. The process repeats every 15 minutes.

This tool provides an intelligent, automated solution for managing tasks in Notion, optimizing scheduling for efficiency and prioritization.
