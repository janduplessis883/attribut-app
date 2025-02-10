from notion_client import Client
import os
from notionhelper import NotionHelper
import requests
from datetime import datetime, timedelta
import streamlit as st
import pandas as pd

current_date = datetime.now()
today_date = current_date.date().isoformat()

database_id = "136fdfd68a9780a3ae4be27f473bad08"

NOTION_ENDPOINT = "https://api.notion.com/v1"
NOTION_DATABASE_ID = database_id  # Replace with your actual Notion Database ID
NOTION_API_TOKEN = st.secrets["NOTION_TOKEN"]  # Replace with your actual token

def notion_api_sync(database_id):
    notion = Client(auth=NOTION_API_TOKEN)

    # Calculate dates
    today = datetime.now().date()
    four_days_later = today + timedelta(days=4)

    # Get tasks based on the updated filter
    response = notion.databases.query(
        database_id=database_id,
        filter={
            "and": [
                {
                    "property": "Status",
                    "status": {
                        "equals": "Not started"
                    }
                },
                {
                    "property": "Deadline",
                    "date": {
                        "on_or_before": four_days_later.isoformat()
                    }
                }
            ]
        },
        page_size=20,
        sorts=[{
            "property": "Deadline",
            "direction": "ascending"
        }]
    )

    tasks = response["results"][:20]  # Limit to 10 tasks

    return tasks


def test_notion():
    nh = NotionHelper()
    result = nh.notion_get_page(database_id)
    return result




def update_task_in_notion(page_id: str, start_datetime: str, end_datetime: str) -> dict:
    """
    Update the start_datetime and end_datetime properties for an existing Task or database page,
    identified by the page_id.

    Args:
        page_id (str): Notion Page ID of the page to be updated or rescheduled.
        start_datetime (str): Start date and time in ISO 8601 format (e.g., "2025-02-04T10:00:00").
        end_datetime (str): End date and time in ISO 8601 format (e.g., "2025-02-04T11:30:00").

    Returns:
        dict: JSON response from the Notion API. If the request fails, returns a dict with 'error' key.
    """
    # Prepare request payload
    task_data = {
        "parent": {"database_id": NOTION_DATABASE_ID},
        "properties": {
            "Scheduled Time": {
                "date": {
                    "start": start_datetime,
                    "end": end_datetime
                }
            }
        },
    }

    # Prepare headers (including the necessary auth and version)
    headers = {
        "Authorization": f"Bearer {NOTION_API_TOKEN}",
        "Notion-Version": "2022-06-28",
        "Content-Type": "application/json"
    }

    # Make the PATCH request to Notion
    url = f"{NOTION_ENDPOINT}/pages/{page_id}"
    response = requests.patch(url, headers=headers, json=task_data)

    # Return the JSON response if successful, or error info if not
    if response.status_code == 200:
        return response.json()
    else:
        return {"error": response.text}


def process_notion_pages(pages, energy):
    """Process Notion API response into structured DataFrame"""
    data = []
    for page in pages:
        props = page.get('properties', {})

        # Safely handle possible missing/None properties
        name_prop = props.get('Name', {}).get('title', [])
        name = name_prop[0]['plain_text'] if name_prop else "Untitled"

        scheduled_time_prop = props.get('Scheduled Time', {}).get('date')
        scheduled_time = scheduled_time_prop.get('start') if scheduled_time_prop else None

        duration = props.get('Duration', {}).get('number', 0)

        priority_prop = props.get('Priority', {}).get('select')
        priority = priority_prop.get('name') if priority_prop else None

        deadline_prop = props.get('Deadline', {}).get('date')
        deadline = deadline_prop.get('start') if deadline_prop else None

        task_data = {
            'Name': name,
            'Duration': duration,
            'Priority': priority,
            'Deadline': deadline,
            'Scheduled Time': scheduled_time,
            'id': page.get('id')
        }

        # Add to session state history if new
        if task_data not in st.session_state.tasks_history:
            st.session_state.tasks_history.append(task_data)
        data.append(task_data)

    df = pd.DataFrame(data)
    datetime_cols = ['Scheduled Time', 'Deadline']
    for col in datetime_cols:
        df[col] = pd.to_datetime(df[col], errors='coerce')  # convert or NaT if invalid
        df[col] = df[col].dt.strftime('%Y-%m-%d %H:%M')
    # Safely sort: if all are NaT, no problem; all will go to bottom

    if energy == 'low':
        df = df.sort_values(by='Duration', ascending=True, na_position='last').reset_index(drop=True)
    elif energy == 'medium':
        df = df.sort_values(by=['Deadline', 'Priority'], ascending=[True, False], na_position='last').reset_index(drop=True)
    elif energy == 'high':
        df = df.sort_values(by=['Deadline', 'Priority'], ascending=[True, True], na_position='last').reset_index(drop=True)

    return df
