from nn import NotionCalendar, Scheduler
import streamlit as st
import time


st.image('images/logo.png')

API_KEY = st.secrets['API_KEY']
DATABASE_ID = st.secrets['DATABASE_ID']

notion_calendar = NotionCalendar(API_KEY, DATABASE_ID)
scheduler = Scheduler(notion_calendar)

if st.checkbox("Scheduler On", value=False):
    # This container will hold our countdown progress bar and text.
    progress_container = st.empty()
    countdown_text = st.empty()

    while True:
        with st.spinner("Scheduling Notion Calendar..."):
            scheduler.schedule_tasks()
        st.write(":material/priority: Successful! / :material/snooze: Sleeping for 15 min")


        # Set countdown for 15 minutes (900 seconds)
        total_seconds = 15 * 60

        # Initialize progress bar
        progress_bar = progress_container.progress(0)

        for elapsed in range(total_seconds):
            # Calculate progress as a float between 0 and 1
            progress = (elapsed + 1) / total_seconds
            progress_bar.progress(progress)

            # Calculate remaining minutes and seconds
            remaining = total_seconds - elapsed - 1
            mins, secs = divmod(remaining, 60)
            countdown_text.text(f"Time remaining: {mins:02d}:{secs:02d}")

            time.sleep(1)
