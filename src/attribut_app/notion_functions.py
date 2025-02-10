import notion_client
from notion_client import Client
from datetime import datetime, timedelta
import json
import pandas as pd

from notionhelper import NotionHelper


def get_all_pages(database_id):
    nh = NotionHelper()
    df = nh.get_all_pages_as_dataframe(database_id)
    return df



def df_to_state(df, start_time, end_time, day_numbers):
    """
    Converts a given DataFrame into a structured JSON-like dictionary
    representing tasks and a calendar.
    """
    import pandas as pd
    from datetime import datetime, timedelta

    # Define priority mapping (assuming 'High' > 'Medium' > 'Low')
    priority_mapping = {"High": 3, "Medium": 2, "Low": 1}

    # Initialize the state structure with updated preferences
    real_world_state = {
        "preferences": {
            "productive_hours": [12, 22],  # Updated productive hours
            "work_days": day_numbers     # Monday to Friday (0 = Monday)
        },
        "tasks": [],
        "calendar": []
    }

    # Convert deadline to datetime, handle missing priorities, and format tasks
    for _, row in df.iterrows():
        # Handle scheduled time (start & end)
        if pd.notna(row["Scheduled Time"]) and row["Scheduled Time"] != "":
            scheduled_start = datetime.fromisoformat(row["Scheduled Time"]).replace(tzinfo=None)
            duration = int(row["Duration"]) if pd.notna(row["Duration"]) else 30
            scheduled_end = scheduled_start + timedelta(minutes=duration)
            scheduled_time = (scheduled_start, scheduled_end)
        else:
            scheduled_time = None

        task = {
            "priority": priority_mapping.get(row["Priority"], 1),  # Default to lowest priority if missing
            "deadline": datetime.strptime(row["Deadline"], "%Y-%m-%d") if pd.notna(row["Deadline"]) else datetime.now(),
            "duration": int(row["Duration"]) if pd.notna(row["Duration"]) else 30,  # Default to 30 mins if missing
            "dependencies": [],
            "type": row["Name"] if pd.notna(row["Name"]) else "Task",
            "scheduled_time": scheduled_time
        }
        real_world_state["tasks"].append(task)

    return real_world_state


# real_world_state = {
#             "preferences": {
#                 "productive_hours": (14, 18)  # for instance, 9 AM to 5 PM
#             },
#             "tasks": [
#                 {
#                     "priority": 1,
#                     "deadline": datetime.now() + timedelta(days=1),
#                     "duration": 15,  # in minutes
#                     "dependencies": [],
#                     "type": "email",
#                     "scheduled_time": (datetime.datetime(2025, 2, 4, 17, 30), datetime.datetime(2025, 2, 4, 18, 30))
#                 },
#                 # ... up to however many tasks you want to simulate
#             ],
#             "calendar": [
#                 {
#                     "start": datetime.now() + timedelta(hours=1),
#                     "end": datetime.now() + timedelta(hours=2)
#                 },
#                 # Additional calendar events as needed.
#             ]
#         }
