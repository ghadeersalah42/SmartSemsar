"""Telegram Bot integration service for SmartSemsar Property Reports.

Provides Telegram Bot API integration to send customer property reports and viewing
appointment confirmations directly to a customer's verified Telegram chat.
"""

from __future__ import annotations

import html
import logging
import os
import secrets
import time
from typing import Any
from pathlib import Path
import requests

from services.property_service import calculate_installments

logger = logging.getLogger(__name__)


def _load_env_file() -> None:
    """Safely parse .env if present without requiring third-party dotenv."""
    env_path = Path(__file__).resolve().parents[2] / ".env"
    if env_path.exists():
        try:
            for line in env_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    k = k.strip()
                    v = v.strip().strip("'\"")
                    if k and k not in os.environ:
                        os.environ[k] = v
        except Exception:
            pass


_load_env_file()

# In-memory registry for one-time expiring Telegram linking tokens
# token -> {"customer_id": str, "created_at": float, "expires_at": float}
_PENDING_LINKING_TOKENS: dict[str, dict[str, Any]] = {}


def get_telegram_bot_token() -> str:
    """Retrieve the Telegram Bot Token from environment variables."""
    return os.getenv("TELEGRAM_BOT_TOKEN", "").strip()


_CACHED_BOT_USERNAME: str | None = None


def fetch_telegram_bot_username(bot_token: str | None = None) -> str | None:
    """Discover verified bot username dynamically from Telegram getMe if available."""
    global _CACHED_BOT_USERNAME
    if _CACHED_BOT_USERNAME:
        return _CACHED_BOT_USERNAME

    token = bot_token or get_telegram_bot_token()
    if not token:
        return None

    try:
        url = f"https://api.telegram.org/bot{token}/getMe"
        resp = requests.get(url, timeout=5)
        data = resp.json()
        if data.get("ok"):
            u = data.get("result", {}).get("username")
            if u:
                _CACHED_BOT_USERNAME = u.lstrip("@")
                return _CACHED_BOT_USERNAME
    except Exception:
        pass
    return None


def get_telegram_bot_username(bot_token: str | None = None) -> str:
    """Retrieve the Telegram Bot Username.

    Prioritizes explicit TELEGRAM_BOT_USERNAME. If set to default placeholder 'SmartSemsarBot'
    or unset in runtime (outside unit tests), auto-discovers the real username from getMe.
    """
    env_user = os.getenv("TELEGRAM_BOT_USERNAME", "").strip().lstrip("@")
    is_testing = bool(os.getenv("PYTEST_CURRENT_TEST"))

    if is_testing:
        return env_user or "SmartSemsarBot"

    if env_user and env_user != "SmartSemsarBot":
        return env_user

    discovered = fetch_telegram_bot_username(bot_token)
    if discovered:
        return discovered

    return env_user or "SmartSemsarBot"


def get_telegram_webhook_info(bot_token: str | None = None) -> dict[str, Any]:
    """Retrieve webhook configuration info safely from Telegram getWebhookInfo.

    Never exposes bot token in error messages.
    """
    token = bot_token or get_telegram_bot_token()
    if not token:
        return {"ok": False, "has_webhook": False, "error": "Telegram Bot Token is not configured."}

    try:
        url = f"https://api.telegram.org/bot{token}/getWebhookInfo"
        resp = requests.get(url, timeout=5)
        data = resp.json()
        if data.get("ok"):
            res = data.get("result")
            if isinstance(res, dict):
                wh_url = res.get("url", "")
                return {
                    "ok": True,
                    "has_webhook": bool(wh_url),
                    "url": wh_url,
                    "pending_update_count": res.get("pending_update_count", 0),
                    "last_error_message": res.get("last_error_message"),
                }
            return {"ok": True, "has_webhook": False, "url": "", "pending_update_count": 0}
        err = _sanitize_secret(data.get("description", "Failed to get webhook info"), token)
        return {"ok": False, "has_webhook": False, "error": err}
    except Exception as exc:
        err = _sanitize_secret(str(exc), token)
        return {"ok": False, "has_webhook": False, "error": f"Network error checking webhook: {err}"}


def is_telegram_configured() -> tuple[bool, str]:
    """Check if the Telegram Bot integration has a configured token."""
    token = get_telegram_bot_token()
    if token:
        return True, "Telegram Bot is configured."
    return False, "Telegram Bot Token is not configured in environment (TELEGRAM_BOT_TOKEN)."


def is_valid_public_url(url: str | None) -> bool:
    """Return True if url is an accessible public HTTP/HTTPS URL.

    Rejects None, empty strings, local file paths, localhost, and relative paths.
    Guarantees zero hallucination and prevents leaking local filesystem paths.
    """
    if not url or not isinstance(url, str):
        return False
    trimmed = url.strip()
    if not (trimmed.startswith("http://") or trimmed.startswith("https://")):
        return False
    lower = trimmed.lower()
    if any(local in lower for local in ["localhost", "127.0.0.1", "0.0.0.0", "::1", "file://"]):
        return False
    # Reject Windows local drive or local path mistakenly prefixed
    if ":\\" in lower or ":/" in lower[7:]:
        return False
    return True


def is_confirmed_booking(booking_data: dict | None, property_id: str | None = None) -> bool:
    """Check if booking_data represents an actually confirmed appointment for the property.

    Never assumes or invents a booking. Requires date, time, and confirmed status or calendar ID.
    """
    if not booking_data or not isinstance(booking_data, dict):
        return False
    if property_id and booking_data.get("property_id") != property_id:
        return False
    if not booking_data.get("date") or not booking_data.get("time"):
        return False

    status = str(booking_data.get("status", "")).lower()
    has_calendar_id = bool(booking_data.get("calendar_event_id"))

    return "confirmed" in status or has_calendar_id


def get_confirmed_booking_for_property(
    property_id: str,
    booking_data: dict | None = None,
    booking_history: list[dict] | None = None,
) -> dict | None:
    """Retrieve the latest confirmed booking matching property_id, if any."""
    if booking_data and is_confirmed_booking(booking_data, property_id):
        return booking_data

    if booking_history and isinstance(booking_history, list):
        for b in reversed(booking_history):
            if is_confirmed_booking(b, property_id):
                return b
    return None


def get_linking_token_info(token: str) -> dict[str, Any] | None:
    """Return active linking token metadata if present and unexpired."""
    if not token or token not in _PENDING_LINKING_TOKENS:
        return None
    info = _PENDING_LINKING_TOKENS[token]
    if time.time() > info.get("expires_at", 0):
        _PENDING_LINKING_TOKENS.pop(token, None)
        return None
    return info


def generate_linking_token(
    customer_id: str = "default",
    ttl_seconds: int = 600,
    bot_token: str | None = None,
) -> dict[str, Any]:
    """Generate a secure, one-time, expiring linking code and Telegram deep-link.

    Default TTL is 10 minutes (600 seconds).
    """
    now = time.time()
    expired = [k for k, v in _PENDING_LINKING_TOKENS.items() if v.get("expires_at", 0) < now]
    for k in expired:
        _PENDING_LINKING_TOKENS.pop(k, None)

    token = f"LINK_{secrets.token_hex(4).upper()}"
    expires_at = now + ttl_seconds

    _PENDING_LINKING_TOKENS[token] = {
        "customer_id": customer_id,
        "created_at": now,
        "expires_at": expires_at,
    }

    bot_username = get_telegram_bot_username(bot_token)
    deep_link = f"https://t.me/{bot_username}?start={token}"

    return {
        "token": token,
        "deep_link": deep_link,
        "expires_at": expires_at,
        "expires_in_seconds": ttl_seconds,
        "bot_username": bot_username,
    }


def _sanitize_secret(text: str, secret: str) -> str:
    """Strip secret token from error messages to avoid accidental log exposure."""
    if secret and secret in text:
        return text.replace(secret, "<REDACTED_BOT_TOKEN>")
    return text


def verify_telegram_linking(token: str, bot_token: str | None = None) -> dict[str, Any]:
    """Verify linking by checking Telegram getUpdates for the customer's /start token command.

    Does not trust arbitrary user-supplied chat IDs: associates chat ID directly from
    Telegram's official update object when the user starts the bot with the linking code.
    Preserves updates queue (never calls getUpdates with offset or deletes updates blindly).
    """
    token = token.strip()
    active_bot_token = bot_token or get_telegram_bot_token()
    bot_username = get_telegram_bot_username(active_bot_token)

    if not token or token not in _PENDING_LINKING_TOKENS:
        return {
            "success": False,
            "error": "Invalid or unknown linking token. Please generate a new connection link.",
        }

    token_info = _PENDING_LINKING_TOKENS[token]
    if time.time() > token_info.get("expires_at", 0):
        _PENDING_LINKING_TOKENS.pop(token, None)
        return {
            "success": False,
            "error": f"This linking code '{token}' has expired. Please generate a new link and try again.",
            "expired": True,
        }

    if not active_bot_token:
        return {
            "success": False,
            "error": "Telegram Bot Token is not configured. Cannot verify account linking with Telegram API.",
        }

    try:
        url = f"https://api.telegram.org/bot{active_bot_token}/getUpdates"
        resp = requests.get(url, timeout=10)
        data = resp.json()
    except Exception as exc:
        err = _sanitize_secret(str(exc), active_bot_token)
        return {
            "success": False,
            "error": f"Failed to connect to Telegram API: {err}",
        }

    if not data.get("ok"):
        raw_desc = data.get("description", "Unknown Telegram API error")
        err_desc = _sanitize_secret(raw_desc, active_bot_token)
        err_lower = err_desc.lower()

        # Handle 409 Conflict: webhook active or concurrent polling
        if "webhook is active" in err_lower or "delete webhook" in err_lower:
            return {
                "success": False,
                "error": (
                    f"Telegram API Conflict: A webhook is currently active on this bot ({err_desc}). "
                    "Telegram disables getUpdates while a webhook is configured. "
                    "Do not delete webhooks blindly if another service relies on them."
                ),
                "conflict": True,
            }
        if "terminated by other getupdates" in err_lower or "conflict" in err_lower:
            return {
                "success": False,
                "error": (
                    f"Telegram API Conflict: Another process is currently polling getUpdates for this bot token ({err_desc}). "
                    "Please ensure only one bot instance/process is running."
                ),
                "conflict": True,
            }
        if data.get("error_code") == 401 or "unauthorized" in err_lower:
            return {
                "success": False,
                "error": f"Telegram API Authentication Error: Invalid or revoked bot token ({err_desc}).",
                "unauthorized": True,
            }

        return {
            "success": False,
            "error": f"Telegram API error: {err_desc}",
        }

    updates = data.get("result", [])
    matched_chat_id = None
    matched_username = ""
    matched_first_name = ""

    token_upper = token.upper()

    # Search for the start message containing our unique token
    for upd in reversed(updates):
        msg = upd.get("message") or upd.get("edited_message") or {}
        msg_text = msg.get("text", "")
        # Accepts e.g., "/start LINK_1234ABCD", "LINK_1234ABCD", or case-insensitive equivalent
        if token_upper in msg_text.upper():
            chat = msg.get("chat", {})
            matched_chat_id = chat.get("id")
            matched_username = chat.get("username", "")
            matched_first_name = chat.get("first_name", "")
            break

    if not matched_chat_id:
        # Check if the user sent a plain /start command or start with missing/mismatched code
        has_plain_start = False
        for upd in reversed(updates):
            msg = upd.get("message") or upd.get("edited_message") or {}
            msg_text = msg.get("text", "").strip()
            if msg_text.startswith("/start"):
                has_plain_start = True
                break

        if has_plain_start:
            return {
                "success": False,
                "error": (
                    f"A start message was detected in Telegram, but it was missing your linking code '{token}'. "
                    f"Please click 'Open Bot in Telegram' to connect automatically, or send '/start {token}' "
                    f"directly in your chat with @{bot_username}."
                ),
                "code_missing": True,
            }

        if not updates:
            return {
                "success": False,
                "error": (
                    "No start message found in Telegram yet. "
                    f"Please click 'Open Bot in Telegram', press 'Start', and then click 'Verify Connection'. "
                    f"(Or send '/start {token}' directly to @{bot_username})"
                ),
            }

        return {
            "success": False,
            "error": (
                f"No matching start command found in recent Telegram updates. "
                f"Please send '/start {token}' to @{bot_username} and click 'Verify Connection'."
            ),
        }

    # Consume the one-time token once successfully matched
    _PENDING_LINKING_TOKENS.pop(token, None)

    # Send a confirmation greeting via Telegram (do not fail linking if greeting send fails)
    confirmation_text = (
        "✅ <b>SmartSemsar Account Linked Successfully!</b>\n\n"
        "Your Telegram chat is now connected to SmartSemsar. "
        "You can now receive comprehensive property reports and viewing appointment updates directly in this chat."
    )
    try:
        send_telegram_message(
            chat_id=matched_chat_id,
            text=confirmation_text,
            parse_mode="HTML",
            bot_token=active_bot_token,
        )
    except Exception as exc:
        logger.warning("Failed to send Telegram greeting message: %s", exc)

    return {
        "success": True,
        "chat_id": str(matched_chat_id),
        "username": matched_username,
        "first_name": matched_first_name,
    }


def format_property_report(
    property_info: dict,
    customer_name: str = "Valued Customer",
    installment_plan: dict | None = None,
    booking_data: dict | None = None,
    format_type: str = "html",
) -> str:
    """Generate a structured, professional property report formatted for Telegram.

    Supports two scenarios:
    1. Without booking: includes property details, price, installment estimate with assumptions.
    2. With confirmed booking: additionally includes verified appointment date and time.

    Formats with safe HTML tags for Telegram API parse_mode="HTML" or plain text.
    """
    name = (customer_name or "").strip() or "Valued Customer"

    prop_id = str(property_info.get("id", "N/A"))
    title = str(property_info.get("title", "Property Details"))
    location = str(property_info.get("location", "N/A"))
    prop_type = str(property_info.get("property_type", "Residential"))
    price = float(property_info.get("price", 0))
    bedrooms = str(property_info.get("bedrooms", "N/A"))
    area_sqm = str(property_info.get("area_sqm", "N/A"))
    finishing = str(property_info.get("finishing", "Standard"))
    description = str(property_info.get("description", ""))

    # Installment plan calculation
    if installment_plan is None:
        dp_pct = 20.0
        years = 5
        interest = 0.0
        inst_data = calculate_installments(price, dp_pct, years, interest)
        inst_data["down_payment_percent"] = dp_pct
        inst_data["years"] = years
        inst_data["annual_interest_rate"] = interest
    else:
        inst_data = installment_plan.copy()
        dp_pct = float(inst_data.get("down_payment_percent", 20.0))
        years = int(inst_data.get("years", 5))
        interest = float(inst_data.get("annual_interest_rate", 0.0))
        if "monthly_installment" not in inst_data:
            calc = calculate_installments(price, dp_pct, years, interest)
            inst_data.update(calc)
        inst_data.setdefault("down_payment_percent", dp_pct)
        inst_data.setdefault("years", years)
        inst_data.setdefault("annual_interest_rate", interest)

    # Verification: only include booking if actually confirmed for this property
    has_confirmed_booking = is_confirmed_booking(booking_data, prop_id)

    # Check for valid public URLs only (zero hallucination, no local paths)
    public_links = []
    link_candidates = [
        ("3D Virtual Tour", property_info.get("public_3d_url") or property_info.get("virtual_tour_url")),
        ("Floor Plan", property_info.get("floor_plan_url")),
        ("Property Image", property_info.get("image_url")),
    ]
    for label, url in link_candidates:
        if is_valid_public_url(url):
            public_links.append((label, url))

    is_html = format_type.lower() == "html"

    def b(txt: str) -> str:
        return f"<b>{html.escape(str(txt))}</b>" if is_html else f"*{txt}*"

    def i(txt: str) -> str:
        return f"<i>{html.escape(str(txt))}</i>" if is_html else f"_{txt}_"

    def link(label_txt: str, url_txt: str) -> str:
        return f'<a href="{url_txt}">{html.escape(label_txt)}</a>' if is_html else f"{label_txt} ({url_txt})"

    def esc(txt: str) -> str:
        return html.escape(str(txt)) if is_html else str(txt)

    lines = [
        f"🏡 {b('SmartSemsar Property Report')}",
        "━━━━━━━━━━━━━━━━━━━━━",
        f"Hello {b(name)}, here is the requested property report:",
        "",
        f"📋 {b('Property Overview')}",
        f"• {b('Property:')} {esc(title)} (ID: {esc(prop_id)})",
        f"• {b('Location:')} 📍 {esc(location)}",
        f"• {b('Type:')} {esc(prop_type)}",
        f"• {b('Price:')} EGP {price:,.0f}",
        f"• {b('Bedrooms:')} 🛏️ {esc(bedrooms)}",
        f"• {b('Area:')} 📐 {esc(area_sqm)} m²",
        f"• {b('Finishing:')} {esc(finishing)}",
    ]

    if description:
        lines.append(f"• {b('Description:')} {esc(description)}")

    lines.extend([
        "",
        f"🧮 {b('Estimated Payment Plan')}",
        f"• {b('Down Payment:')} EGP {inst_data.get('down_payment', 0):,.0f} ({inst_data.get('down_payment_percent', 0):.0f}%)",
        f"• {b('Duration:')} {inst_data.get('years', 0)} years",
        f"• {b('Monthly Installment:')} EGP {inst_data.get('monthly_installment', 0):,.0f} / month",
        f"• {b('Remaining Amount:')} EGP {inst_data.get('remaining', 0):,.0f}",
        f"• {b('Total Interest:')} EGP {inst_data.get('total_interest', 0):,.0f}",
        f"• {b('Estimated Total Cost:')} EGP {inst_data.get('total_cost', 0):,.0f}",
        f"• {b('Interest Assumption:')} {inst_data.get('annual_interest_rate', 0):.1f}% flat annual rate",
        f"ℹ️ {i('Note: This estimate uses simple flat interest on the remaining balance; actual developer payment plans may differ.')}",
    ])

    if has_confirmed_booking and booking_data is not None:
        b_date = booking_data.get("date", "N/A")
        b_time = booking_data.get("time", "N/A")
        lines.extend([
            "",
            f"📅 {b('Confirmed Viewing Appointment')}",
            f"• {b('Date:')} {esc(b_date)}",
            f"• {b('Time:')} {esc(b_time)}",
            f"• {b('Status:')} Confirmed viewing appointment",
        ])
        if booking_data.get("notes"):
            lines.append(f"• {b('Notes:')} {esc(booking_data['notes'])}")
        if is_valid_public_url(booking_data.get("calendar_link")):
            lines.append(f"• {b('Google Calendar:')} {link('Open in Calendar', booking_data['calendar_link'])}")

    if public_links:
        lines.extend([
            "",
            f"🌐 {b('Interactive Assets & Tours')}",
        ])
        for lbl, u in public_links:
            lines.append(f"• {b(lbl + ':')} {link('Open Link', u)}")

    lines.extend([
        "",
        "━━━━━━━━━━━━━━━━━━━━━",
        f"{i('Thank you for choosing SmartSemsar! Let us know if you would like to adjust payment options or arrange additional viewings.')}",
    ])

    return "\n".join(lines)


def send_telegram_message(
    chat_id: int | str,
    text: str,
    parse_mode: str = "HTML",
    bot_token: str | None = None,
) -> dict[str, Any]:
    """Send an arbitrary message to a Telegram chat via Telegram Bot API sendMessage.

    Returns dict with success=True and message details, or success=False and sanitized error.
    Never exposes or logs the bot token.
    """
    if not chat_id:
        return {"success": False, "error": "Telegram chat ID is missing."}

    token = bot_token or get_telegram_bot_token()
    if not token:
        return {"success": False, "error": "Telegram Bot Token is not configured."}

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": parse_mode,
        "disable_web_page_preview": False,
    }

    try:
        resp = requests.post(url, json=payload, timeout=12)
        data = resp.json()
    except Exception as exc:
        err = _sanitize_secret(str(exc), token)
        return {"success": False, "error": f"Network error sending Telegram message: {err}"}

    if not data.get("ok"):
        desc = _sanitize_secret(data.get("description", "Unknown Telegram API error"), token)
        return {"success": False, "error": f"Telegram API error: {desc}"}

    result = data.get("result", {})
    return {
        "success": True,
        "message_id": result.get("message_id"),
        "chat_id": chat_id,
        "date": result.get("date"),
    }


def send_property_report(
    chat_id: int | str,
    property_info: dict,
    customer_name: str = "Valued Customer",
    installment_plan: dict | None = None,
    booking_data: dict | None = None,
    bot_token: str | None = None,
) -> dict[str, Any]:
    """Format and send a property report directly to a customer's Telegram chat.

    Handles both Scenario 1 (without booking) and Scenario 2 (with verified booking).
    Reports success only if the Telegram API confirms delivery.
    """
    if not chat_id:
        return {
            "success": False,
            "error": "Telegram chat ID is missing. Please link your Telegram account first.",
        }

    formatted_text = format_property_report(
        property_info=property_info,
        customer_name=customer_name,
        installment_plan=installment_plan,
        booking_data=booking_data,
        format_type="html",
    )

    send_res = send_telegram_message(
        chat_id=chat_id,
        text=formatted_text,
        parse_mode="HTML",
        bot_token=bot_token,
    )

    if send_res.get("success"):
        return {
            "success": True,
            "message_id": send_res.get("message_id"),
            "chat_id": str(chat_id),
            "report_text": formatted_text,
        }
    return {
        "success": False,
        "error": send_res.get("error", "Unknown error sending property report."),
        "report_text": formatted_text,
    }
