import datetime
import requests
import json
import time
from tqdm import tqdm

# ---------------------------
# Task Model
# ---------------------------
class Task:
    def __init__(self, id, title, duration, deadline, scheduled_time, priority, status):
        """
        Initialize a task.
        - duration: in minutes.
        - deadline and scheduled_time are expected as ISO format strings or datetime objects.
        """
        self.id = id
        self.title = title
        self.duration = duration  # in minutes
        # Convert string representations to datetime objects if necessary
        self.deadline = self._to_datetime(deadline)
        self.scheduled_time = self._to_datetime(scheduled_time)
        self.priority = priority
        self.status = status
        self.end_time = self.scheduled_time + datetime.timedelta(minutes=self.duration)

    def _to_datetime(self, dt):
        if isinstance(dt, str):
            # Handle timestamps ending with 'Z'
            if dt.endswith("Z"):
                dt = dt.replace("Z", "+00:00")
            # Parse and then remove timezone info to keep it naive
            return datetime.datetime.fromisoformat(dt).replace(tzinfo=None)
        return dt.replace(tzinfo=None)

    def update_schedule(self, new_start):
        """Update the scheduled start time and recalculate the end time."""
        self.scheduled_time = new_start
        self.end_time = self.scheduled_time + datetime.timedelta(minutes=self.duration)

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
                "Status": {"status": {"name": self.status}}
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

        # Payload with filter: Only tasks with Status "Not started" and Deadline on or before four days from now.
        payload = {
            "filter": {
                "and": [
                    {
                        "property": "Status",
                        "status": {"equals": "Not started"}
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

                task_id = result.get("id")

                # Only create tasks if both deadline and scheduled_time are present.
                if deadline and scheduled_time:
                    task = Task(task_id, title, duration, deadline, scheduled_time, priority, status)
                    tasks.append(task)
                else:
                    print(f"❗️ Skipping task {task_id} due to missing deadline or scheduled_time.")
        else:
            print("Error fetching tasks:", response.text)

        return tasks

    def update_task(self, task):
        """Update a task’s scheduling information in Notion."""
        url = f"{self.base_url}/pages/{task.id}"
        payload = task.to_notion_format()
        response = requests.patch(url, headers=self.headers, data=json.dumps(payload))
        if response.status_code != 200:
            print(f"⛔️ Error updating task {task.id}: {response.text}")
        else:
            print(f"✅ Task '{task.title}' (ID: {task.id}) updated successfully.")

# ---------------------------
# Scheduler
# ---------------------------
class Scheduler:
    def __init__(self, notion_calendar):
        self.notion_calendar = notion_calendar

    def _get_next_working_time(self, dt):
        """
        Adjust dt to the next available working time:
          - Working hours: 09:00 to 16:00.
          - Only Monday to Friday.
        """
        # If dt is before 09:00, set to 09:00 of that day.
        if dt.hour < 9:
            dt = dt.replace(hour=9, minute=0, second=0, microsecond=0)
        # If dt is at or after 16:00, push to the next day at 09:00.
        elif dt.hour >= 16:
            dt = dt + datetime.timedelta(days=1)
            dt = dt.replace(hour=9, minute=0, second=0, microsecond=0)
        # If dt falls on a weekend, push to the next Monday at 09:00.
        while dt.weekday() >= 5:
            dt = dt + datetime.timedelta(days=1)
            dt = dt.replace(hour=9, minute=0, second=0, microsecond=0)
        return dt

    def schedule_tasks(self):
        """
        Fetch tasks and reschedule those with status "Not started".
        Tasks are ordered such that urgent ones (deadline ≤ now + 1 hour) are scheduled first (by deadline),
        then the rest are ordered by descending priority.
        Tasks are scheduled sequentially during working hours (09:00–16:00).
        If a task’s duration doesn't fit in the current day, it’s moved to the next working day.
        """
        tasks = self.notion_calendar.get_tasks()
        now = datetime.datetime.now()

        # Filter only tasks with status "Not started" (for rescheduling).
        tasks = [task for task in tasks if task.status.lower() == "not started"]

        # Separate into two groups:
        group_a = [task for task in tasks if task.deadline <= now + datetime.timedelta(hours=1)]
        group_b = [task for task in tasks if task.deadline > now + datetime.timedelta(hours=1)]
        group_a.sort(key=lambda t: t.deadline)   # Urgent: earlier deadlines first.
        group_b.sort(key=lambda t: -t.priority)    # Then by high priority.
        tasks = group_a + group_b

        # Start scheduling from the next available working time.
        current_time = self._get_next_working_time(now)
        print(f"🟠 Rescheduling tasks starting at {current_time}")

        for task in tasks:
            # Determine the end of the working day.
            working_end = current_time.replace(hour=16, minute=0, second=0, microsecond=0)
            # If the task cannot be completed before working hours end, push current_time to the next working day.
            if current_time + datetime.timedelta(minutes=task.duration) > working_end:
                current_time = self._get_next_working_time(current_time + datetime.timedelta(days=1))
                working_end = current_time.replace(hour=16, minute=0, second=0, microsecond=0)

            # Schedule the task at the current_time.
            task.update_schedule(current_time)
            print(f"Scheduling '{task.title}' at {task.scheduled_time} until {task.end_time}")
            # Update current_time to the end of the scheduled task.
            current_time = task.end_time
            # If current_time exceeds working hours, adjust to next working time.
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

    print("🏁 Starting scheduler loop. Rescheduling tasks every 15 minutes...")
    while True:
        scheduler.schedule_tasks()
        # Wait for 15 minutes before rescheduling.
        sleep_duration = 15 * 60  # 5 minutes
        print(f"💤 Sleeping for {sleep_duration} seconds.")
        for i in tqdm(range(sleep_duration), desc="Waiting until next reschedule", unit="sec"):
            time.sleep(1)
