import datetime
import requests
import json
import time
from tqdm import tqdm

# ---------------------------
# Colorama Setup
# ---------------------------
from colorama import init, Fore, Style
init(autoreset=True)

# Define color constants:
NN_COLOR = Fore.LIGHTBLACK_EX      # For all NN events (predictions, training loss, simulated targets)
SCHEDULE_COLOR = Fore.LIGHTYELLOW_EX # For scheduling events (approximating orange)
SUCCESS_COLOR = Fore.LIGHTGREEN_EX   # For tasks updated successfully
ERROR_COLOR = Fore.LIGHTRED_EX       # For errors or warnings

# ---------------------------
# For the Neural Network
# ---------------------------
import torch
import torch.nn as nn
import torch.optim as optim

class RescheduleNN(nn.Module):
    def __init__(self, input_size=4, hidden_size1=16, hidden_size2=16, hidden_size3=8, output_size=1):
        super(RescheduleNN, self).__init__()
        # First hidden layer: from input_size to hidden_size1 neurons
        self.fc1 = nn.Linear(input_size, hidden_size1)
        # Second hidden layer: from hidden_size1 to hidden_size2 neurons
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        # Third hidden layer: from hidden_size2 to hidden_size3 neurons
        self.fc3 = nn.Linear(hidden_size2, hidden_size3)
        # Output layer: from hidden_size3 to output_size (predicts extra time in minutes)
        self.fc4 = nn.Linear(hidden_size3, output_size)

        # ReLU activation to introduce non-linearity
        self.relu = nn.ReLU()

    def forward(self, x):
        # Pass through first hidden layer and apply ReLU
        x = self.relu(self.fc1(x))
        # Pass through second hidden layer and apply ReLU
        x = self.relu(self.fc2(x))
        # Pass through third hidden layer and apply ReLU
        x = self.relu(self.fc3(x))
        # Pass through the output layer
        x = self.fc4(x)
        # Apply ReLU again to ensure non-negative output
        return self.relu(x)

# ---------------------------
# Task Model
# ---------------------------
class Task:
    def __init__(self, id, title, duration, deadline, scheduled_time, priority, status, reschedule_count=0, fixed_schedule=False):
        """
        Initialize a task.
        - duration: in minutes.
        - deadline and scheduled_time are expected as ISO format strings or datetime objects.
        - reschedule_count: tracks how many times the task has been rescheduled.
        - fixed_schedule: if True, the scheduled date remains fixed and won't be updated.
        """
        self.id = id
        self.title = title
        self.duration = duration  # in minutes
        # Convert string representations to datetime objects if necessary
        self.deadline = self._to_datetime(deadline)
        self.scheduled_time = self._to_datetime(scheduled_time)
        self.priority = priority
        self.status = status
        self.reschedule_count = reschedule_count
        self.fixed_schedule = fixed_schedule  # New attribute for fixed scheduling
        self.end_time = self.scheduled_time + datetime.timedelta(minutes=self.duration)

    def _to_datetime(self, dt):
        if isinstance(dt, str):
            # Handle timestamps ending with 'Z'
            if dt.endswith("Z"):
                dt = dt.replace("Z", "+00:00")
            # Parse and then remove timezone info to keep it naive
            return datetime.datetime.fromisoformat(dt).replace(tzinfo=None)
        return dt.replace(tzinfo=None)

    def update_schedule(self, new_start, extra_time=0):
        """Update the scheduled start time and recalculate the end time.
           The extra_time (in minutes) is added to the task duration to account for underestimated durations.
        """
        self.scheduled_time = new_start
        effective_duration = self.duration + extra_time
        self.end_time = self.scheduled_time + datetime.timedelta(minutes=effective_duration)
        # Increment the reschedule count so that our NN sees history
        self.reschedule_count += 1

    def to_notion_format(self):
        """
        Prepare the task properties in the format expected by the Notion API.
        Assumes the database uses a specific property schema.
        """
        return {
            "properties": {
                "Title": {
                    "title": [{"text": {"content": self.title}}]
                },
                "Duration": {"number": self.duration},
                "Deadline": {"date": {"start": self.deadline.isoformat()}},
                "Scheduled Time": {
                    "date": {
                        "start": self.scheduled_time.isoformat(),
                        "end": self.end_time.isoformat()
                    }
                },
                # Send Priority as a select type (convert the number to a string)
                "Priority": {"select": {"name": str(self.priority)}},
                # Send Status as a status type
                "Status": {"status": {"name": self.status}},
                # Preserve the Fixed Schedule property
                "Fixed Schedule": {"checkbox": self.fixed_schedule}
            }
        }

# ---------------------------
# Notion Calendar Integration
# ---------------------------
class NotionCalendar:
    def __init__(self, api_key, database_id):
        self.api_key = api_key
        self.database_id = database_id
        self.base_url = "https://api.notion.com/v1"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Notion-Version": "2022-06-28",
            "Content-Type": "application/json"
        }

    def get_tasks(self):
        """Fetch tasks from the Notion Calendar Database with specific filters."""
        url = f"{self.base_url}/databases/{self.database_id}/query"

        # Define the date four days from now
        four_days_later = datetime.datetime.now() + datetime.timedelta(days=4)

        # Payload with filter: tasks with Status "Not started" OR "In progress"
        # and Deadline on or before four days from now.
        payload = {
            "filter": {
                "and": [
                    {
                        "or": [
                            {"property": "Status", "status": {"equals": "Not started"}},
                            {"property": "Status", "status": {"equals": "In progress"}}
                        ]
                    },
                    {
                        "property": "Deadline",
                        "date": {"on_or_before": four_days_later.isoformat()}
                    }
                ]
            },
            "page_size": 30,
            "sorts": [{
                "property": "Deadline",
                "direction": "ascending"
            }]
        }

        response = requests.post(url, headers=self.headers, data=json.dumps(payload))
        tasks = []

        if response.status_code == 200:
            data = response.json()
            for result in data.get("results", []):
                properties = result.get("properties", {})

                # Extract Title (assume at least one text fragment exists)
                title_fragments = properties.get("Title", {}).get("title", [])
                title = title_fragments[0]["text"]["content"] if title_fragments else "Untitled"

                duration = properties.get("Duration", {}).get("number", 0)

                # Safely extract Deadline
                deadline_prop = properties.get("Deadline", {}).get("date")
                deadline = deadline_prop.get("start") if deadline_prop else None

                # Safely extract Scheduled Time
                scheduled_time_prop = properties.get("Scheduled Time", {}).get("date")
                scheduled_time = scheduled_time_prop.get("start") if scheduled_time_prop else None

                # Extract Priority from the select property and convert to integer for sorting.
                priority_prop = properties.get("Priority", {}).get("select", {})
                priority_str = priority_prop.get("name", "0")
                try:
                    priority = int(priority_str)
                except ValueError:
                    priority = 0

                # Extract Status from the status property.
                status = properties.get("Status", {}).get("status", {}).get("name", "pending")

                # Extract the Fixed Schedule checkbox property.
                fixed_schedule = properties.get("Fixed Schedule", {}).get("checkbox", False)

                task_id = result.get("id")

                # Only create tasks if both deadline and scheduled_time are present.
                if deadline and scheduled_time:
                    task = Task(task_id, title, duration, deadline, scheduled_time, priority, status,
                                fixed_schedule=fixed_schedule)
                    tasks.append(task)
                else:
                    print(ERROR_COLOR + f"❗️ Skipping task {task_id} due to missing deadline or scheduled_time." + Style.RESET_ALL)
        else:
            print(ERROR_COLOR + "Error fetching tasks:" + response.text + Style.RESET_ALL)

        return tasks

    def update_task(self, task):
        """Update a task’s scheduling information in Notion."""
        url = f"{self.base_url}/pages/{task.id}"
        payload = task.to_notion_format()
        response = requests.patch(url, headers=self.headers, data=json.dumps(payload))
        if response.status_code != 200:
            print(ERROR_COLOR + f"⛔️ Error updating task {task.id}: {response.text}" + Style.RESET_ALL)
        else:
            print(SUCCESS_COLOR + f"✅ Task '{task.title}' (ID: {task.id}) updated successfully.\n" + Style.RESET_ALL)

# ---------------------------
# Scheduler with NN Integration
# ---------------------------
class Scheduler:
    def __init__(self, notion_calendar):
        self.notion_calendar = notion_calendar

        # Initialize the NN model, optimizer, and loss function.
        self.nn_model = RescheduleNN()
        self.optimizer = optim.Adam(self.nn_model.parameters(), lr=0.01)
        self.criterion = nn.MSELoss()

    def _get_next_working_time(self, dt):
        """
        Adjust dt to the next available working time:
          - Working hours: 09:00 to 16:00.
          - Only Monday to Friday.
        """
        if dt.hour < 9:
            dt = dt.replace(hour=9, minute=0, second=0, microsecond=0)
        elif dt.hour >= 16:
            dt = dt + datetime.timedelta(days=1)
            dt = dt.replace(hour=9, minute=0, second=0, microsecond=0)
        while dt.weekday() >= 5:
            dt = dt + datetime.timedelta(days=1)
            dt = dt.replace(hour=9, minute=0, second=0, microsecond=0)
        return dt

    def get_next_fixed_interval(self, current_time, tasks):
        """
        Returns the next fixed schedule interval (start, end) from the list of tasks
        where 'Fixed Schedule' is True and whose end time is after the current_time.
        If current_time falls within a fixed interval, that interval is returned.
        """
        fixed_intervals = [
            (t.scheduled_time, t.end_time)
            for t in tasks
            if t.fixed_schedule and t.end_time > current_time
        ]
        if not fixed_intervals:
            return None
        fixed_intervals.sort(key=lambda x: x[0])
        for interval in fixed_intervals:
            # If current_time is within an interval, return it.
            if interval[0] <= current_time < interval[1]:
                return interval
            # Otherwise, if the interval starts after current_time, return the earliest one.
            if interval[0] > current_time:
                return interval
        return None

    def schedule_tasks(self):
        """
        Fetch tasks and reschedule those with status "Not started" or "In progress".
        For tasks in progress, update the start time to the current time and extend the end time.
        Additionally, non-fixed tasks will not be scheduled during fixed schedule timeslots.
        """
        tasks = self.notion_calendar.get_tasks()
        now = datetime.datetime.now()

        # Filter tasks by status "Not started" or "In progress"
        tasks = [task for task in tasks if task.status.lower() in ["not started", "in progress"]]

        # Separate urgent tasks (deadline ≤ now + 1 hour) from the rest.
        group_a = [task for task in tasks if task.deadline <= now + datetime.timedelta(hours=1)]
        group_b = [task for task in tasks if task.deadline > now + datetime.timedelta(hours=1)]
        group_a.sort(key=lambda t: t.deadline)   # Urgent: earlier deadlines first.
        group_b.sort(key=lambda t: -t.priority)    # Then by high priority.
        tasks = group_a + group_b

        # Start scheduling from the next available working time.
        current_time = self._get_next_working_time(now)
        print(SCHEDULE_COLOR + f"🟠 Rescheduling tasks starting at {current_time}" + Style.RESET_ALL)

        margin = 5  # Safety margin in minutes

        for task in tasks:
            # Skip tasks that are fixed; they already have their timeslots reserved.
            if task.fixed_schedule:
                print(SCHEDULE_COLOR + f"🔒 Task '{task.title}' is marked as fixed schedule. Skipping rescheduling." + Style.RESET_ALL)
                continue

            # Prepare features for the NN:
            minutes_since_midnight = current_time.hour * 60 + current_time.minute
            time_norm = minutes_since_midnight / 1440.0
            features = torch.tensor([[float(task.reschedule_count),
                                       float(task.priority),
                                       float(task.duration),
                                       time_norm]], dtype=torch.float32)
            predicted_extra_time_tensor = self.nn_model(features)
            predicted_extra_time = predicted_extra_time_tensor.item()
            print(NN_COLOR + f"🤖 NN predicts extra time of {predicted_extra_time:.2f} min for task '{task.title}'" + Style.RESET_ALL)

            # --- Simulated Target for NN Training ---
            simulated_factor = 1 + 0.05 * task.reschedule_count
            time_factor = 1 + ((current_time.hour - 9) * 0.01) if current_time.hour > 9 else 1
            priority_factor = max(0.8, 1 - task.priority * 0.01)
            effective_factor = simulated_factor * time_factor * priority_factor
            target_extra_time = max(0, task.duration * (effective_factor - 1))
            print(NN_COLOR + f"🎯 Simulated target extra time: {target_extra_time:.2f} min for task '{task.title}'" + Style.RESET_ALL)

            target_tensor = torch.tensor([[target_extra_time]], dtype=torch.float32)
            loss = self.criterion(predicted_extra_time_tensor, target_tensor)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            print(NN_COLOR + f"🔧 Training loss: {loss.item():.4f}" + Style.RESET_ALL)

            # --- Handling tasks based on their status ---
            if task.status.lower() == "in progress":
                new_start_time = current_time
                old_remaining = max(0, (task.end_time - current_time).total_seconds() / 60)
                effective_remaining = old_remaining + predicted_extra_time
                new_end_time = new_start_time + datetime.timedelta(minutes=effective_remaining)
                print(SCHEDULE_COLOR + f"⏰ Updating in-progress task '{task.title}':" + Style.RESET_ALL)
                print(SCHEDULE_COLOR + f"    • Old end time: {task.end_time}" + Style.RESET_ALL)
                print(SCHEDULE_COLOR + f"    • New start time: {new_start_time}" + Style.RESET_ALL)
                print(SCHEDULE_COLOR + f"    • Extended end time: {new_end_time}" + Style.RESET_ALL)
                task.scheduled_time = new_start_time
                task.end_time = new_end_time
                task.reschedule_count += 1
                if task.end_time > task.deadline:
                    print(ERROR_COLOR + f"⛔️ Warning: In-progress task '{task.title}' now ends at {task.end_time} which is after its deadline {task.deadline}." + Style.RESET_ALL)
                self.notion_calendar.update_task(task)
                continue

            # Ensure the task fits within working hours.
            working_end = current_time.replace(hour=16, minute=0, second=0, microsecond=0)
            if current_time + datetime.timedelta(minutes=task.duration) > working_end:
                current_time = self._get_next_working_time(current_time + datetime.timedelta(days=1))
                working_end = current_time.replace(hour=16, minute=0, second=0, microsecond=0)

            effective_duration = task.duration + predicted_extra_time

            # **** NEW LOGIC: Adjust current_time to avoid fixed schedule intervals ****
            while True:
                next_fixed = self.get_next_fixed_interval(current_time, tasks)
                if not next_fixed:
                    break
                fixed_start, fixed_end = next_fixed
                candidate_end = current_time + datetime.timedelta(minutes=effective_duration)
                # If current_time is inside a fixed interval, shift to its end.
                if fixed_start <= current_time < fixed_end:
                    print(SCHEDULE_COLOR + f"🔒 Current time {current_time} is within fixed interval {fixed_start} to {fixed_end}. Shifting start time." + Style.RESET_ALL)
                    current_time = fixed_end
                    continue
                # If the candidate task would overlap a fixed interval, shift start time to after that fixed interval.
                if candidate_end > fixed_start:
                    print(SCHEDULE_COLOR + f"🔒 Candidate end time {candidate_end} overlaps fixed interval starting at {fixed_start}. Shifting start time to {fixed_end}." + Style.RESET_ALL)
                    current_time = fixed_end
                    continue
                break

            # Update task schedule with the (possibly adjusted) current_time.
            task.update_schedule(current_time, extra_time=predicted_extra_time)
            print(SCHEDULE_COLOR + f"⏰ Scheduling '{task.title}' at {task.scheduled_time} until {task.end_time}" + Style.RESET_ALL)
            current_time = task.end_time
            current_time = self._get_next_working_time(current_time)
            self.notion_calendar.update_task(task)

# ---------------------------
# Main Loop for Rescheduling
# ---------------------------
if __name__ == "__main__":
    # Replace with your actual Notion API key and Database ID.
    API_KEY = "secret_AUqFdk1kzS6qe7iw0LVlPDQXJ1TrDxnM7n9ZIB5fOlB"
    DATABASE_ID = "136fdfd68a9780a3ae4be27f473bad08"

    notion_calendar = NotionCalendar(API_KEY, DATABASE_ID)
    scheduler = Scheduler(notion_calendar)

    print(SCHEDULE_COLOR + "🏁 Starting scheduler loop. Rescheduling tasks every 15 minutes..." + Style.RESET_ALL)
    while True:
        scheduler.schedule_tasks()
        # Wait for 15 minutes before rescheduling.
        sleep_duration = 15 * 60  # 15 minutes in seconds
        print(SCHEDULE_COLOR + f"💤 Sleeping for {sleep_duration} seconds." + Style.RESET_ALL)
        for i in tqdm(range(sleep_duration), desc="Waiting until next reschedule", unit="sec"):
            time.sleep(1)
