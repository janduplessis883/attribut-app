import threading
import time
import io
import contextlib
import queue
import streamlit as st
from streamlit_autorefresh import st_autorefresh

# Import the Scheduler and NotionCalendar classes from your nn.py file.
# (Ensure nn.py is in the same directory as this app.py.)
from nn import Scheduler, NotionCalendar

# ------------------------------
# Global Variables (shared between runs)
# ------------------------------
# We use module-level globals so that the background thread can update them,
# and the main Streamlit app can read them on each rerun.
if "scheduler_thread" not in globals():
    scheduler_thread = None
if "stop_event" not in globals():
    stop_event = threading.Event()
if "log_list" not in globals():
    log_list = []
if "current_countdown" not in globals():
    current_countdown = 0

# ------------------------------
# Scheduler Loop Function
# ------------------------------
def scheduler_loop(api_key, database_id):
    """
    Runs a continuous loop where:
      - It calls scheduler.schedule_tasks() (which prints its progress)
      - Captures all print output and appends it to a global log list
      - Then waits for 15 minutes with a countdown (updating the global countdown)
    This loop stops when the stop_event is set.
    """
    global current_count, log_list, current_countdown  # use globals
    # Initialize NotionCalendar and Scheduler from the backend
    notion_calendar = NotionCalendar(api_key, database_id)
    scheduler = Scheduler(notion_calendar)

    while not stop_event.is_set():
        # Capture output from one scheduling cycle
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            scheduler.schedule_tasks()
        output = buf.getvalue()
        for line in output.splitlines():
            log_list.append(line)

        # Wait for 15 minutes (900 seconds), updating the countdown each second.
        for sec in range(900, 0, -1):
            if stop_event.is_set():
                break
            current_countdown = sec
            time.sleep(1)
    # Reset the countdown when stopped.
    current_countdown = 0

# ------------------------------
# Streamlit App Layout
# ------------------------------
st.title("Neural Network Scheduler Front End")
st.write("This app lets you switch on the scheduling process and shows live progress and logs.")

# Let the user provide their Notion API key and Database ID.
API_KEY = st.text_input("Notion API Key", value="secret_AUqFdk1kzS6qe7iw0LVlPDQXJ1TrDxnM7n9ZIB5fOlB")
DATABASE_ID = st.text_input("Notion Database ID", value="136fdfd68a9780a3ae4be27f473bad08")

# A checkbox to enable/disable the scheduling process.
enable_sched = st.checkbox("Enable Scheduling", value=False)

# Start/stop the background scheduler thread based on the checkbox state.
if enable_sched:
    if scheduler_thread is None or not scheduler_thread.is_alive():
        # Clear any previous stop signal
        stop_event.clear()
        scheduler_thread = threading.Thread(
            target=scheduler_loop, args=(API_KEY, DATABASE_ID), daemon=True
        )
        scheduler_thread.start()
        st.success("Scheduler started!")
else:
    # If the checkbox is unchecked and the scheduler is running, signal it to stop.
    if scheduler_thread is not None and scheduler_thread.is_alive():
        stop_event.set()
        scheduler_thread.join(timeout=1)
        st.info("Scheduler stopped.")

# ------------------------------
# Display Progress Bar for Countdown
# ------------------------------
# Compute the progress as a fraction.
progress_fraction = (900 - current_countdown) / 900 if current_countdown else 0
st.subheader("Waiting Time Until Next Scheduling Cycle")
progress_bar = st.progress(progress_fraction)
st.write(f"Next scheduling in **{current_countdown}** seconds.")

# ------------------------------
# Display Logs from the Scheduler
# ------------------------------
st.subheader("Scheduler Logs")
# Join all log lines into a single text block.
logs_text = "\n".join(log_list)
st.text_area("Logs", logs_text, height=300)

# ------------------------------
# Auto-refresh the app every second so the UI stays up-to-date
# ------------------------------
st_autorefresh(interval=1000, key="scheduler_autorefresh")
