import streamlit as st
import pandas as pd
from datetime import datetime, timedelta, time as datetime_time
import time
from supabase import create_client


# -- Import your Notion & RL logic
from live_notion_api import notion_api_sync, update_task_in_notion, process_notion_pages

from utils import get_next_working_day
# --- Supabase setup ---
supabase_url = st.secrets["SUPABASE_URL"]
supabase_key = st.secrets["SUPABASE_KEY"]
supabase = create_client(supabase_url, supabase_key)

# --- Streamlit page config ---
st.set_page_config("Attribut AI", page_icon=":material/today:", layout="wide")

# --- Initialize session state ---
if "notion_sync_frequency" not in st.session_state:
    st.session_state.notion_sync_frequency = 900  # 10 minutes in seconds
if "next_call" not in st.session_state:
    st.session_state.next_call = time.time() + st.session_state.notion_sync_frequency
# Initialize tasks_history if it's missing
if "tasks_history" not in st.session_state:
    st.session_state.tasks_history = []
# Keep references to tasks, probabilities, and model states for feedback
if "last_ranked_tasks" not in st.session_state:
    st.session_state.last_ranked_tasks = []
if "last_probs" not in st.session_state:
    st.session_state.last_probs = None
if "last_state_tensor" not in st.session_state:
    st.session_state.last_state_tensor = None
if "last_converted_tasks" not in st.session_state:
    st.session_state.last_converted_tasks = []
if "energy" not in st.session_state:
    st.session_state.energy = 'high'
if "work_days" not in st.session_state:
    st.session_state.work_days = ['Mon', 'Tues', 'Wed', 'Thurs']


database_id = "136fdfd68a9780a3ae4be27f473bad08"
current_datetime = datetime.now()

# --- Title & Branding ---
st.title(":material/today: Attribut AI")
st.logo("images/logo.png", size="medium")

# --- Sidebar: toggle + settings ---
sync_toggle = st.sidebar.toggle(":material/data_object: **Sync with Notion** (10 min)", value=True)
st.sidebar.container(height=20, border=False)
st.sidebar.header(":material/settings: Settings")

with st.sidebar.popover(":material/power_settings_new: Notion API Setting"):
    notion_token_store = st.text_input("Notion Token")
    database_id_store = st.text_input("Database ID")

with st.sidebar.popover(":material/engineering: Working Pattern Setup"):
    options = ["Mon", "Tues", "Wed", "Thurs", "Fri", "Sat", "Sun"]
    selection = st.segmented_control("Work Days", options, default=['Mon', 'Tues', 'Wed', 'Thurs'], selection_mode="multi")
    day_mapping = {"Mon": 0, "Tues": 1, "Wed": 2, "Thurs": 3, "Fri": 4, "Sat": 5, "Sun": 6}

    # Create a list of 7 zeros (one for each day, Mon–Sun)
    days_selected = [0] * 7
    # Mark the selected days with 1
    for day in selection:
        idx = day_mapping[day]
        days_selected[idx] = 1

    start_time, end_time = st.slider(
        "Select a time range",
        0.0, 24.0, (9.5, 16.0),
        step=0.5,
        format="%.1f"
    )
    st.write(
        f"**Productive Hours**: "
        f"{int(start_time):02d}:{int((start_time % 1)*60):02d} - "
        f"{int(end_time):02d}:{int((end_time % 1)*60):02d}"
    )
    energy = st.radio("Energy Level", ['low', 'medium', 'high'], index=2, horizontal=True)

    st.write(f"Next Working Day: {get_next_working_day()}")

st.sidebar.container(height=30, border=False)

def show_countdown_timer():
    """Display and manage the countdown timer."""
    current_time = time.time()
    remaining = st.session_state.next_call - current_time

    progress_bar = st.sidebar.progress(0)
    time_placeholder = st.sidebar.empty()

    # This loop blocks the app; consider st_autorefresh for non-blocking
    while remaining > 0:
        mins, secs = divmod(int(remaining), 60)
        time_str = f"Next update in: **{mins:02d}:{secs:02d}**"
        progress = 1 - (remaining / float(st.session_state.notion_sync_frequency))

        progress_bar.progress(progress)
        time_placeholder.markdown(f"**{time_str}**")
        time.sleep(1)
        remaining = st.session_state.next_call - time.time()

    progress_bar.progress(1.0)
    time_placeholder.markdown("**Updating schedule now...**")

    st.session_state.next_call = time.time() + st.session_state.notion_sync_frequency
    st.rerun()

def schedule_tasks(tasks_list):
    """
    Schedules tasks sequentially within working hours (9:00 to 16:00).
    - If run before 9:00, scheduling starts at 9:00.
    - If run after 16:00, scheduling starts the next working day at 9:00.
    - If run during working hours, scheduling starts from the current time.
    Each task is scheduled for its `Duration` (in minutes) and a 2-minute buffer is added after each task.
    Optionally, tasks that already have a 'start' time (and are thus in progress or complete) are skipped.
    """
    now = datetime.now()

    # Determine the starting point for scheduling
    if now.time() < datetime_time(9, 0):
        rolling_datetime = now.replace(hour=9, minute=0, second=0, microsecond=0)
    elif now.time() >= datetime_time(16, 0):
        rolling_datetime = get_next_working_day(now)
    else:
        rolling_datetime = now

    for task in tasks_list:
        # Optional: Skip tasks that already have a scheduled start time
        if 'start' in task:
            scheduled_start = datetime.fromisoformat(task['start'])
            # If the task is already in progress or has started, skip it.
            if scheduled_start <= now:
                continue

        # If our rolling time is at or past 16:00, jump to the next working day at 9:00.
        if rolling_datetime.time() >= datetime_time(16, 0):
            rolling_datetime = get_next_working_day(rolling_datetime)

        # Calculate how many minutes remain until 16:00 on the current day.
        day_end = rolling_datetime.replace(hour=16, minute=0, second=0, microsecond=0)
        available_minutes = (day_end - rolling_datetime).total_seconds() / 60

        # If the current task's duration doesn't fit before 16:00,
        # move rolling_datetime to the next working day at 9:00.
        if available_minutes < task["Duration"]:
            rolling_datetime = get_next_working_day(rolling_datetime)
            day_end = rolling_datetime.replace(hour=16, minute=0, second=0, microsecond=0)

        # Schedule the task.
        start_datetime = rolling_datetime
        end_datetime = start_datetime + timedelta(minutes=task["Duration"])

        with st.spinner(f"Scheduling task with Notion API call..."):
            update_task_in_notion(
                task["id"],
                start_datetime.isoformat(),
                end_datetime.isoformat()
            )

        # Add a 2-minute buffer before scheduling the next task.
        rolling_datetime = end_datetime + timedelta(minutes=2)
        time.sleep(0.3)

    show_countdown_timer()

# -----------------------------------------
# Main sync logic
if sync_toggle:
    with st.spinner("Fetching Notion data..."):
        pages = notion_api_sync(database_id)
        df = process_notion_pages(pages, energy=energy)

        st.dataframe(df)

    # Convert DataFrame to list of dicts
    tasks_dict = df.to_dict(orient="records")


    schedule_tasks(tasks_dict)



# -------------------------------------------------
# USER FEEDBACK SECTION

st.header("Notion Sync OFF")
