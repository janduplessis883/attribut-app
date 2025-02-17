import datetime
import requests
import json

# ---------------------------
# Task Model
# ---------------------------
class Task:
    def __init__(self, id, title, duration, deadline, scheduled_time, priority, status):
        self.id = id
        self.title = title
        self.duration = duration  # in minutes
        self.deadline = self._to_datetime(deadline)
        self.scheduled_time = self._to_datetime(scheduled_time)
        self.priority = priority
        self.status = status
        self.end_time = self.scheduled_time + datetime.timedelta(minutes=self.duration)

    def _to_datetime(self, dt):
        if isinstance(dt, str):
            if dt.endswith("Z"):
                dt = dt.replace("Z", "+00:00")
            return datetime.datetime.fromisoformat(dt).replace(tzinfo=None)
        return dt.replace(tzinfo=None)

    def update_schedule(self, new_start):
        self.scheduled_time = new_start
        self.end_time = self.scheduled_time + datetime.timedelta(minutes=self.duration)

    def to_notion_format(self):
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
                "Priority": {"select": {"name": str(self.priority)}},
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
        url = f"{self.base_url}/databases/{self.database_id}/query"
        four_days_later = datetime.datetime.now() + datetime.timedelta(days=4)
        payload = {
            "filter": {
                "and": [
                    {"property": "Status", "status": {"equals": "Not started"}},
                    {"property": "Deadline", "date": {"on_or_before": four_days_later.isoformat()}}
                ]
            },
            "page_size": 30,
            "sorts": [{"property": "Deadline", "direction": "ascending"}]
        }
        response = requests.post(url, headers=self.headers, data=json.dumps(payload))
        tasks = []
        if response.status_code == 200:
            data = response.json()
            for result in data.get("results", []):
                properties = result.get("properties", {})
                title_fragments = properties.get("Title", {}).get("title", [])
                title = title_fragments[0]["text"]["content"] if title_fragments else "Untitled"
                duration = properties.get("Duration", {}).get("number", 0)
                deadline_prop = properties.get("Deadline", {}).get("date")
                deadline = deadline_prop.get("start") if deadline_prop else None
                scheduled_time_prop = properties.get("Scheduled Time", {}).get("date")
                scheduled_time = scheduled_time_prop.get("start") if scheduled_time_prop else None
                priority_prop = properties.get("Priority", {}).get("select", {})
                priority_str = priority_prop.get("name", "0")
                try:
                    priority = int(priority_str)
                except ValueError:
                    priority = 0
                status = properties.get("Status", {}).get("status", {}).get("name", "pending")
                task_id = result.get("id")
                if deadline and scheduled_time:
                    task = Task(task_id, title, duration, deadline, scheduled_time, priority, status)
                    tasks.append(task)
                else:
                    print(f"❗️ Skipping task {task_id} due to missing deadline or scheduled_time.")
        else:
            print("Error fetching tasks:", response.text)
        return tasks

    def update_task(self, task):
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
        if dt.hour < 9:
            dt = dt.replace(hour=9, minute=0, second=0, microsecond=0)
        elif dt.hour >= 16:
            dt = dt + datetime.timedelta(days=1)
            dt = dt.replace(hour=9, minute=0, second=0, microsecond=0)
        while dt.weekday() >= 5:
            dt = dt + datetime.timedelta(days=1)
            dt = dt.replace(hour=9, minute=0, second=0, microsecond=0)
        return dt

    def schedule_tasks(self):
        tasks = self.notion_calendar.get_tasks()
        now = datetime.datetime.now()
        tasks = [task for task in tasks if task.status.lower() == "not started"]
        group_a = [task for task in tasks if task.deadline <= now + datetime.timedelta(hours=1)]
        group_b = [task for task in tasks if task.deadline > now + datetime.timedelta(hours=1)]
        group_a.sort(key=lambda t: t.deadline)
        group_b.sort(key=lambda t: -t.priority)
        tasks = group_a + group_b
        current_time = self._get_next_working_time(now)
        print(f"🟠 Rescheduling tasks starting at {current_time}")
        for task in tasks:
            working_end = current_time.replace(hour=16, minute=0, second=0, microsecond=0)
            if current_time + datetime.timedelta(minutes=task.duration) > working_end:
                current_time = self._get_next_working_time(current_time + datetime.timedelta(days=1))
                working_end = current_time.replace(hour=16, minute=0, second=0, microsecond=0)
            task.update_schedule(current_time)
            print(f"Scheduling '{task.title}' at {task.scheduled_time} until {task.end_time}")
            current_time = task.end_time
            current_time = self._get_next_working_time(current_time)
            self.notion_calendar.update_task(task)
