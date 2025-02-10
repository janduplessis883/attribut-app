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

# -----------------------------------------
# Main sync logic
if sync_toggle:
    with st.spinner("Fetching Notion data..."):
        pages = notion_api_sync(database_id)
        df = process_notion_pages(pages, energy=energy)

        st.dataframe(df)

    # Convert DataFrame to list of dicts
    tasks_dict = df.to_dict(orient="records")
    st.code(tasks_dict)


    # Example scheduling logi
    for task in tasks_dict:
        rolling_datetime = datetime.now()
        if rolling_datetime.time() >= datetime_time(16, 0):
            rolling_datetime = get_next_working_day()
        else:
            # Start today but clamp to 9:30 if needed
            if rolling_datetime.time() < datetime_time(9, 00):
                rolling_datetime = rolling_datetime.replace(hour=9, minute=0, second=0, microsecond=0)

        for task in tasks_dict:
            # Check if current time is outside working hours
            if rolling_datetime.time() >= datetime_time(16, 0):
                rolling_datetime = get_next_working_day(rolling_datetime)

            # Calculate maximum available time today
            day_end = rolling_datetime.replace(hour=16, minute=0, second=0)
            available_minutes = (day_end - rolling_datetime).total_seconds() / 60

            # Check if task fits in current day
            if available_minutes < task["Duration"]:
                rolling_datetime = get_next_working_day(rolling_datetime)
                day_end = rolling_datetime.replace(hour=16, minute=0)

            # Schedule task
            start_datetime = rolling_datetime
            end_datetime = start_datetime + timedelta(minutes=task["Duration"])

            # Update Notion
            with st.spinner("Updating Notion via API..."):
                update_task_in_notion(
                    task["id"],
                    start_datetime.isoformat(),
                    end_datetime.isoformat()
                )

            # Set next slot with 2-minute buffer
            rolling_datetime = end_datetime + timedelta(minutes=2)
            time.sleep(0.3)





    # Show the countdown timer
    show_countdown_timer()

# -------------------------------------------------
# USER FEEDBACK SECTION
st.markdown("---")
st.header("User Feedback (RL Reward)")

def show_feedback_table():
    # If we have tasks from the last sync
    if not st.session_state.last_ranked_tasks:
        st.info("No tasks available for feedback. Please sync with Notion first.")
        return

    # Build a DataFrame from the last-ranked tasks
    df_ranked = pd.DataFrame(st.session_state.last_ranked_tasks)

    # Add a "User Feedback" column with default "No Feedback"
    if "User Feedback" not in df_ranked.columns:
        df_ranked["User Feedback"] = "No Feedback"

    # Define column config so "User Feedback" is a selectbox with 3 radio options
    column_config = {
        "User Feedback": st.column_config.SelectboxColumn(
            label="User Feedback",
            options=["No Feedback", "Positive", "Neutral", "Negative"]
        )
    }

    edited_df = st.data_editor(
        df_ranked,
        use_container_width=True,
        num_rows="fixed",  # don't allow adding new rows
        key="feedback_editor",
        column_config=column_config
    )

    st.caption("Select 'Positive', 'Neutral', or 'Negative' in the 'User Feedback' column for each task.")

    if st.button("Submit Feedback"):
        # Retrieve relevant RL data
        if st.session_state.last_probs is None or st.session_state.last_state_tensor is None:
            st.warning("No model state found to store transitions.")
            return

        state_tensor = st.session_state.last_state_tensor
        probs_tensor = st.session_state.last_probs  # shape [1, 10] typically

        feedback_count = 0

        for idx, row in edited_df.iterrows():
            feedback_val = row.get("User Feedback", "No Feedback")
            if feedback_val == "No Feedback":
                continue  # skip storing a transition

            # Convert feedback_val to a numeric reward
            if feedback_val == "Positive":
                reward = 2.0
            elif feedback_val == "Negative":
                reward = -3.0
            else:  # "Neutral"
                reward = 0.0

            # The row index 'idx' hopefully matches our model action index,
            # if the ordering is consistent with the sorted tasks. If we changed ordering,
            # we'd need a more robust mapping.
            chosen_action_idx = idx  # or some mapping

            # old_prob (model probability for that action)
            # We clamp it if idx >= 10 or if tasks < 10
            if idx < probs_tensor.shape[1]:
                old_prob = probs_tensor[0, chosen_action_idx].unsqueeze(0)  # shape [1]
            else:
                # If user had more tasks than the model input or an invalid index,
                # skip or set old_prob=0
                continue

            # Store in PPO memory
            task_id = row.get("Task ID", f"Unknown_{idx}")
            st.session_state.trainer.store_transition(
                state=state_tensor,
                action=chosen_action_idx,
                old_prob=old_prob,
                reward=reward,
                task_id=task_id
            )
            feedback_count += 1

        st.success(f"Feedback stored for {feedback_count} tasks! You can now update the model below.")

# Call the function to show feedback table
show_feedback_table()

if st.button("Update Model (PPO Step)"):
    loss_value = trainer.update_policy()
    if loss_value is not None:
        st.write(f"**Model updated**. Loss: {loss_value:.4f}")
    else:
        st.write("No transitions to update from.")
