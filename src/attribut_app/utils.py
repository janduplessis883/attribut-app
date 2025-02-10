from datetime import datetime, timedelta, time as datetime_time
import time

def get_next_working_day(start_date=None):
    """
    Get the next working day at 9:00 AM, skipping weekends,
    used when it’s past 16:00 or not enough time remains in the day.
    """
    if start_date is None:
        start_date = datetime.now()

    # Move to the next day
    next_day = start_date + timedelta(days=1)

    # Skip weekends (Saturday=5, Sunday=6)
    while next_day.weekday() in (5, 6):
        next_day += timedelta(days=1)

    # Set time to 09:00
    next_day = next_day.replace(hour=9, minute=0, second=0, microsecond=0)
    return next_day
