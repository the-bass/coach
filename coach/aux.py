import time
import datetime


def minutes_since(reference_datetime):
    now = datetime.datetime.now()
    difference = now - reference_datetime
    difference_in_seconds = difference.total_seconds()
    difference_in_minutes = difference_in_seconds / 60.0

    return difference_in_minutes

def seconds_since(reference_time):
    return time.time() - reference_time
