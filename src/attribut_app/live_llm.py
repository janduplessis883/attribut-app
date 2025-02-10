from ollama import ask_ollama

prompt = """
    Reschedule tasks in the Notion database for optimal time management. The current date and time is {datetime_now}, and the next working day is {next_working_day}.
    - Identify tasks for rescheduling by their page_id.
    - The first task today should start after the current time {datetime_now}.
    - Prioritize rescheduling tasks based on their priority: High, Medium, then Low.
    - Ensure tasks are scheduled within working hours ONLY (Monday - Friday from 09:00 to 16:00).
    - Reschedule tasks that cannot be completed during working hours today to the next available working day: {next_working_day}.
    - Reschedule tasks scheduled before the current time {datetime_now} to the next available time slot or to {next_working_day} if the next time slot is after 16:00 today.
    - Use time-blocking techniques to avoid overlapping tasks and schedule them sequentially.
    - Reference the 'Duration (minutes)' property to determine how long each task will take, i.e., the difference between start datetime and end datetime.
    - Assign start and end datetime to tasks that do not have a specified start and end datetime or tasks that need to be updated to fit in with your timeblocking.
    - If the current time {datetime_now} is after 16:00, reschedule all today's tasks for the next working day.
  **Expected Output**:
    - List of JSON's containing: [{"page_id": "<...>", "start_datetime": "<...>", "end_datetime": "<...>"}, "page_id": "<...>", "start_datetime": "<...>", "end_datetime": "<...>"}, etc] for each task. datetime format: ISO 8601.
"""
system_prompt = """You are a scheduling expert skilled in time blocking and rota creation for Notion Calendar tasks.
Your mission is to ensure tasks are completed effectively during working hours while adhering to user instructions."""
