"""Google Calendar integration service for SmartSemsar.

Provides OAuth 2.0 authentication, event creation, and appointment fetching
using the official Google Calendar API.
"""

from __future__ import annotations

import datetime
import json
import logging
import os
from pathlib import Path
from typing import Any

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import Resource, build
from googleapiclient.errors import HttpError

logger = logging.getLogger(__name__)

# Scopes required to create and read viewing appointments
SCOPES = ["https://www.googleapis.com/auth/calendar.events"]

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def get_credentials_path() -> Path:
    """Resolve the path to the Google OAuth credentials JSON file."""
    env_path = os.getenv("GOOGLE_CALENDAR_CREDENTIALS_FILE", "").strip()
    if env_path:
        p = Path(env_path)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        return p

    candidates = [
        PROJECT_ROOT / "credentials.json",
        PROJECT_ROOT / "smart_semsar_ui" / "credentials.json",
        Path.cwd() / "credentials.json",
    ]
    for c in candidates:
        if c.exists():
            return c
    return PROJECT_ROOT / "credentials.json"


def get_token_path() -> Path:
    """Resolve the path to the saved Google OAuth token JSON file."""
    env_path = os.getenv("GOOGLE_CALENDAR_TOKEN_FILE", "").strip()
    if env_path:
        p = Path(env_path)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        return p

    candidates = [
        PROJECT_ROOT / "token.json",
        PROJECT_ROOT / "smart_semsar_ui" / "token.json",
        Path.cwd() / "token.json",
    ]
    for c in candidates:
        if c.exists():
            return c
    return PROJECT_ROOT / "token.json"


def get_calendar_id() -> str:
    """Return the configured Google Calendar ID (default: 'primary')."""
    return os.getenv("GOOGLE_CALENDAR_ID", "primary").strip() or "primary"


def is_google_calendar_configured() -> tuple[bool, str]:
    """Check if Google Calendar credentials or tokens are configured.

    Returns:
        tuple[bool, str]: (is_configured, reason_or_status)
    """
    token_file = get_token_path()
    if token_file.exists():
        return True, f"Token file found at {token_file.name}."

    creds_file = get_credentials_path()
    if creds_file.exists():
        return True, f"Credentials file found at {creds_file.name}."

    # Direct client ID / secret / refresh token via environment
    if os.getenv("GOOGLE_CALENDAR_CLIENT_ID") and os.getenv("GOOGLE_CALENDAR_CLIENT_SECRET"):
        return True, "Configured via environment variables."

    # Raw JSON credentials string via environment
    if os.getenv("GOOGLE_CALENDAR_CREDENTIALS_JSON"):
        return True, "Configured via GOOGLE_CALENDAR_CREDENTIALS_JSON."

    return False, "Google Calendar credentials file (credentials.json) or environment variables not found."


def is_google_calendar_authenticated() -> bool:
    """Check whether a valid or refreshable OAuth token is currently available."""
    token_file = get_token_path()
    if token_file.exists():
        creds = get_credentials(interactive=False)
        return creds is not None and creds.valid
    if os.getenv("GOOGLE_CALENDAR_REFRESH_TOKEN"):
        creds = get_credentials(interactive=False)
        return creds is not None and creds.valid
    return False


def authorize_google_calendar(interactive: bool = True) -> tuple[bool, str]:
    """Launch the OAuth 2.0 consent flow or refresh existing credentials.

    Args:
        interactive: If True, launches the local web server to authorize in browser.

    Returns:
        tuple[bool, str]: (success, message)
    """
    token_file = get_token_path()
    creds_file = get_credentials_path()

    # 1. If token already exists and is valid, return success
    if token_file.exists():
        creds = get_credentials(interactive=False)
        if creds and creds.valid:
            return True, f"Google Calendar is already authorized. Token loaded from {token_file.name}."

    # 2. Check if credentials file or env config is present
    has_creds = creds_file.exists() or os.getenv("GOOGLE_CALENDAR_CREDENTIALS_JSON") or (
        os.getenv("GOOGLE_CALENDAR_CLIENT_ID") and os.getenv("GOOGLE_CALENDAR_CLIENT_SECRET")
    )
    if not has_creds:
        return False, "Google credentials file (credentials.json) not found in project root or configured path."

    # 3. Launch interactive flow or refresh
    try:
        creds = get_credentials(interactive=interactive)
        if creds and creds.valid:
            return True, f"Google Calendar authorized successfully. Token saved to {token_file.name}."
        return False, "Google Calendar authorization was not completed. Please try again."
    except Exception as e:
        logger.error("Authorization failed: %s", type(e).__name__)
        return False, f"Authorization error: {type(e).__name__} - {str(e)}"


def get_credentials(interactive: bool = False) -> Credentials | None:
    """Load or refresh Google OAuth credentials.

    Args:
        interactive: If True, allows launching local browser OAuth flow when needed.

    Returns:
        Credentials object if authenticated, None otherwise.
    """
    creds: Credentials | None = None
    token_file = get_token_path()
    creds_file = get_credentials_path()

    # 1. Try loading from token file
    if token_file.exists():
        try:
            creds = Credentials.from_authorized_user_file(str(token_file), SCOPES)
        except Exception as e:
            logger.warning("Failed to load existing token file: %s", type(e).__name__)
            creds = None

    # 2. Try loading from environment refresh token
    if not creds:
        client_id = os.getenv("GOOGLE_CALENDAR_CLIENT_ID")
        client_secret = os.getenv("GOOGLE_CALENDAR_CLIENT_SECRET")
        refresh_token = os.getenv("GOOGLE_CALENDAR_REFRESH_TOKEN")
        if client_id and client_secret and refresh_token:
            try:
                creds = Credentials(
                    token=None,
                    refresh_token=refresh_token,
                    token_uri="https://oauth2.googleapis.com/token",
                    client_id=client_id,
                    client_secret=client_secret,
                    scopes=SCOPES,
                )
            except Exception as e:
                logger.warning("Failed to create credentials from env: %s", type(e).__name__)

    # 3. Refresh expired credentials
    if creds and creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
            # Save refreshed credentials
            try:
                token_file.parent.mkdir(parents=True, exist_ok=True)
                token_file.write_text(creds.to_json(), encoding="utf-8")
            except Exception as save_err:
                logger.warning("Could not persist refreshed token: %s", type(save_err).__name__)
        except Exception as ref_err:
            logger.warning("Token refresh failed: %s", type(ref_err).__name__)
            creds = None

    # 4. If still invalid and interactive mode is requested, run OAuth flow
    if (not creds or not creds.valid) and interactive:
        flow: InstalledAppFlow | None = None
        if creds_file.exists():
            flow = InstalledAppFlow.from_client_secrets_file(str(creds_file), SCOPES)
        elif os.getenv("GOOGLE_CALENDAR_CREDENTIALS_JSON"):
            client_config = json.loads(os.getenv("GOOGLE_CALENDAR_CREDENTIALS_JSON", "{}"))
            flow = InstalledAppFlow.from_client_config(client_config, SCOPES)
        elif os.getenv("GOOGLE_CALENDAR_CLIENT_ID") and os.getenv("GOOGLE_CALENDAR_CLIENT_SECRET"):
            client_config = {
                "installed": {
                    "client_id": os.getenv("GOOGLE_CALENDAR_CLIENT_ID"),
                    "client_secret": os.getenv("GOOGLE_CALENDAR_CLIENT_SECRET"),
                    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                    "token_uri": "https://oauth2.googleapis.com/token",
                }
            }
            flow = InstalledAppFlow.from_client_config(client_config, SCOPES)

        if flow:
            try:
                creds = flow.run_local_server(port=0, prompt="consent", access_type="offline")
                if creds:
                    token_file.parent.mkdir(parents=True, exist_ok=True)
                    token_file.write_text(creds.to_json(), encoding="utf-8")
            except Exception as flow_err:
                logger.error("OAuth interactive authorization failed: %s", type(flow_err).__name__)
                creds = None

    return creds if (creds and creds.valid) else None


def get_calendar_service(
    credentials: Credentials | None = None,
    interactive: bool = False,
) -> Resource:
    """Build and return an authorized Google Calendar API service Resource."""
    creds = credentials or get_credentials(interactive=interactive)
    if not creds:
        raise ValueError(
            "Google Calendar is not authorized. Please authorize access to generate token.json."
        )
    return build("calendar", "v3", credentials=creds, cache_discovery=False)



def build_event_data(
    property_info: dict[str, Any],
    customer_name: str,
    customer_phone: str,
    appointment_date: str | datetime.date,
    appointment_time: str,
    notes: str = "",
    duration_minutes: int = 60,
    timezone: str = "Africa/Cairo",
) -> dict[str, Any]:
    """Construct an event payload according to Google Calendar API specification.

    Args:
        property_info: Property dictionary with id, title, location, price, etc.
        customer_name: Full name of customer.
        customer_phone: Contact phone number.
        appointment_date: Date as ISO string ('YYYY-MM-DD') or datetime.date.
        appointment_time: Time as 'HH:MM' string (e.g. '14:00').
        notes: Additional appointment or viewing notes.
        duration_minutes: Duration of the viewing in minutes (default 60).
        timezone: Timezone for the event (default 'Africa/Cairo').

    Returns:
        dict: Google Calendar event resource dictionary.
    """
    if isinstance(appointment_date, str):
        parsed_date = datetime.date.fromisoformat(appointment_date.strip())
    else:
        parsed_date = appointment_date

    time_str = appointment_time.strip()
    hour, minute = [int(p) for p in time_str.split(":")[:2]]
    parsed_time = datetime.time(hour=hour, minute=minute)

    start_dt = datetime.datetime.combine(parsed_date, parsed_time)
    end_dt = start_dt + datetime.timedelta(minutes=duration_minutes)

    prop_id = str(property_info.get("id", "N/A"))
    prop_title = str(property_info.get("title", "Property Viewing"))
    prop_loc = str(property_info.get("location", ""))
    prop_price = property_info.get("price", 0)

    summary = f"Property Viewing: {prop_title} ({prop_id}) - {customer_name}"

    description_parts = [
        "🏡 SmartSemsar Viewing Appointment",
        "",
        "Customer Information:",
        f"• Name: {customer_name}",
        f"• Phone: {customer_phone}",
        "",
        "Property Information:",
        f"• ID: {prop_id}",
        f"• Title: {prop_title}",
        f"• Location: {prop_loc}",
        f"• Price: EGP {prop_price:,.0f}" if isinstance(prop_price, (int, float)) else f"• Price: {prop_price}",
    ]

    bedrooms = property_info.get("bedrooms")
    if bedrooms is not None:
        description_parts.append(f"• Bedrooms: {bedrooms}")

    area_sqm = property_info.get("area_sqm")
    if area_sqm is not None:
        description_parts.append(f"• Area: {area_sqm} m²")

    clean_notes = notes.strip()
    description_parts.extend([
        "",
        "Viewing Notes:",
        clean_notes if clean_notes else "No specific notes provided.",
        "",
        "Automated booking via SmartSemsar.",
    ])

    return {
        "summary": summary,
        "location": prop_loc,
        "description": "\n".join(description_parts),
        "start": {
            "dateTime": start_dt.isoformat(),
            "timeZone": timezone,
        },
        "end": {
            "dateTime": end_dt.isoformat(),
            "timeZone": timezone,
        },
        "reminders": {
            "useDefault": False,
            "overrides": [
                {"method": "popup", "minutes": 60},
                {"method": "popup", "minutes": 1440},  # 24 hours
            ],
        },
        "extendedProperties": {
            "private": {
                "source": "SmartSemsar",
                "property_id": prop_id,
                "customer_name": customer_name,
                "customer_phone": customer_phone,
            }
        },
    }


def create_viewing_event(
    property_info: dict[str, Any],
    customer_name: str,
    customer_phone: str,
    appointment_date: str | datetime.date,
    appointment_time: str,
    notes: str = "",
    duration_minutes: int = 60,
    timezone: str = "Africa/Cairo",
    calendar_id: str | None = None,
    service: Resource | None = None,
) -> dict[str, Any]:
    """Create a viewing event in Google Calendar.

    Args:
        property_info: Property dictionary.
        customer_name: Customer's name.
        customer_phone: Customer's phone number.
        appointment_date: Date of appointment.
        appointment_time: Time of appointment ('HH:MM').
        notes: Viewing notes.
        duration_minutes: Event duration in minutes.
        timezone: Calendar timezone.
        calendar_id: Optional calendar ID (defaults to environment or 'primary').
        service: Optional pre-built Google Calendar Resource for testing/dependency injection.

    Returns:
        dict: Result object containing:
            - 'success': bool
            - 'event_id': str (if success)
            - 'html_link': str (if success)
            - 'event': dict (if success)
            - 'error': str (if failure)
    """
    try:
        cal_service = service or get_calendar_service()
    except Exception as auth_err:
        return {
            "success": False,
            "error": f"Authentication error: {str(auth_err)}",
        }

    event_payload = build_event_data(
        property_info=property_info,
        customer_name=customer_name,
        customer_phone=customer_phone,
        appointment_date=appointment_date,
        appointment_time=appointment_time,
        notes=notes,
        duration_minutes=duration_minutes,
        timezone=timezone,
    )

    target_cal_id = calendar_id or get_calendar_id()

    try:
        created_event = cal_service.events().insert(
            calendarId=target_cal_id,
            body=event_payload,
        ).execute()

        event_id = created_event.get("id")
        html_link = created_event.get("htmlLink")

        return {
            "success": True,
            "event_id": event_id,
            "html_link": html_link,
            "event": created_event,
        }
    except HttpError as http_err:
        error_msg = f"Google Calendar API HTTP {http_err.resp.status}: {http_err.error_details or http_err.reason}"
        logger.error("Failed to insert event: %s", error_msg)
        return {
            "success": False,
            "error": error_msg,
        }
    except Exception as err:
        error_msg = f"Unexpected error creating event: {str(err)}"
        logger.error("Error creating event: %s", error_msg)
        return {
            "success": False,
            "error": error_msg,
        }


def get_upcoming_events(
    max_results: int = 10,
    calendar_id: str | None = None,
    service: Resource | None = None,
) -> dict[str, Any]:
    """Fetch upcoming viewing appointments from Google Calendar.

    Args:
        max_results: Maximum events to return.
        calendar_id: Optional target calendar ID.
        service: Optional pre-built Resource.

    Returns:
        dict: Result object containing:
            - 'success': bool
            - 'events': list[dict]
            - 'error': str (if failure)
    """
    try:
        cal_service = service or get_calendar_service()
    except Exception as auth_err:
        return {
            "success": False,
            "error": f"Authentication error: {str(auth_err)}",
            "events": [],
        }

    target_cal_id = calendar_id or get_calendar_id()
    now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()

    try:
        events_result = cal_service.events().list(
            calendarId=target_cal_id,
            timeMin=now_iso,
            maxResults=max_results,
            singleEvents=True,
            orderBy="startTime",
        ).execute()

        items = events_result.get("items", [])
        parsed_events = []
        for item in items:
            start_info = item.get("start", {})
            start_val = start_info.get("dateTime") or start_info.get("date", "")
            end_info = item.get("end", {})
            end_val = end_info.get("dateTime") or end_info.get("date", "")

            private_props = item.get("extendedProperties", {}).get("private", {})

            parsed_events.append({
                "id": item.get("id"),
                "summary": item.get("summary", "Viewing Appointment"),
                "start": start_val,
                "end": end_val,
                "location": item.get("location", ""),
                "description": item.get("description", ""),
                "html_link": item.get("htmlLink", ""),
                "property_id": private_props.get("property_id"),
                "customer_name": private_props.get("customer_name"),
                "customer_phone": private_props.get("customer_phone"),
            })

        return {
            "success": True,
            "events": parsed_events,
        }
    except HttpError as http_err:
        error_msg = f"Google Calendar API HTTP {http_err.resp.status}: {http_err.reason}"
        logger.error("Failed to list events: %s", error_msg)
        return {
            "success": False,
            "error": error_msg,
            "events": [],
        }
    except Exception as err:
        error_msg = f"Unexpected error listing events: {str(err)}"
        logger.error("Error listing events: %s", error_msg)
        return {
            "success": False,
            "error": error_msg,
            "events": [],
        }


if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("SmartSemsar - Google Calendar OAuth Setup")
    print("=" * 60)

    configured, status = is_google_calendar_configured()
    print(f"Status: {status}")

    if not configured:
        print("\nPlease place your OAuth Desktop client credentials JSON as 'credentials.json'")
        print("in the project root or set GOOGLE_CALENDAR_CREDENTIALS_FILE in .env.")
        sys.exit(1)

    print("\nLaunching Google OAuth consent in your default browser...")
    print("Please select your Google account and grant Calendar permissions.\n")
    success, message = authorize_google_calendar(interactive=True)
    if success:
        print(f"[SUCCESS] {message}")
        print("\nYou can now start the SmartSemsar application to schedule viewings.")
    else:
        print(f"[ERROR] {message}")
        sys.exit(1)
