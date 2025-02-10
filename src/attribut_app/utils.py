from datetime import datetime, timedelta, time as datetime_time

def get_next_working_day(start_date=None):
    """Get next working day at 9:30 AM, skipping weekends and after 16:00 cutoff"""
    if start_date is None:
        start_date = datetime.now()

    current_date = start_date + timedelta(days=1)  # Always move to next day
    # Skip weekends
    while current_date.weekday() in (5, 6):  # Saturday=5, Sunday=6
        current_date += timedelta(days=1)
    # Set time to 09:30 AM
    return current_date.replace(hour=9, minute=0, second=0, microsecond=0)
