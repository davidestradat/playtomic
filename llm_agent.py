"""
LLM Agent for Playtomic Club Manager

Uses OpenAI GPT-4o with function calling to answer natural language
questions about club occupancy, revenue, members, and operations.
"""

import json
from datetime import datetime, timedelta, date, timezone
from typing import Optional
from zoneinfo import ZoneInfo

import openai
import pandas as pd

from playtomic_api import PlaytomicAPI

# ── Timezone helpers ────────────────────────────────────────────────────

_club_tz: ZoneInfo = ZoneInfo("America/Cancun")


def set_club_timezone(tz_name: str):
    """Set the club's timezone for all conversions."""
    global _club_tz
    _club_tz = ZoneInfo(tz_name)


def _utc_to_local_dt(utc_str: str) -> Optional[datetime]:
    """Convert a UTC datetime string to a timezone-aware local datetime object."""
    if not utc_str or "T" not in utc_str:
        return None
    try:
        dt_utc = datetime.strptime(utc_str[:19], "%Y-%m-%dT%H:%M:%S").replace(
            tzinfo=timezone.utc
        )
        return dt_utc.astimezone(_club_tz)
    except (ValueError, IndexError):
        return None


def _utc_to_local(utc_str: str) -> str:
    """Convert a UTC datetime string (YYYY-MM-DDTHH:MM:SS) to local ISO string."""
    dt = _utc_to_local_dt(utc_str)
    return dt.strftime("%Y-%m-%dT%H:%M:%S") if dt else utc_str


def _utc_to_local_date(utc_str: str) -> str:
    """Convert a UTC datetime string to local date string (YYYY-MM-DD)."""
    dt = _utc_to_local_dt(utc_str)
    return dt.strftime("%Y-%m-%d") if dt else utc_str[:10] if utc_str else ""


def _utc_to_readable_time(utc_str: str) -> str:
    """Convert UTC datetime to human-readable local time like '7:00 PM'."""
    dt = _utc_to_local_dt(utc_str)
    if not dt:
        return utc_str
    # Format as 7:00 PM (no leading zero, strip trailing :00 seconds)
    return dt.strftime("%-I:%M %p")


def _utc_to_local_hour(utc_str: str) -> int:
    """Convert UTC datetime to local hour (0-23) for aggregation."""
    dt = _utc_to_local_dt(utc_str)
    return dt.hour if dt else 0


# ── Tool definitions for OpenAI function calling ────────────────────────

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_occupancy_for_date",
            "description": (
                "Get court occupancy and booking details for a specific date. "
                "Shows how busy the club is, which courts are booked, and available slots. "
                "Use this when the user asks about how busy the club is on a specific day."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "date": {
                        "type": "string",
                        "description": "The date to check in YYYY-MM-DD format",
                    },
                },
                "required": ["date"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_occupancy_for_range",
            "description": (
                "Get court occupancy summary across multiple days. "
                "Use this when the user asks about a week, weekend, or range of dates."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "start_date": {
                        "type": "string",
                        "description": "Start date in YYYY-MM-DD format",
                    },
                    "end_date": {
                        "type": "string",
                        "description": "End date in YYYY-MM-DD format",
                    },
                },
                "required": ["start_date", "end_date"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_revenue_summary",
            "description": (
                "Get revenue data for a date range. Shows total revenue, "
                "bookings by payment status, average booking value, and revenue by court. "
                "Use this when the user asks about revenue, income, or financial performance."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "start_date": {
                        "type": "string",
                        "description": "Start date in YYYY-MM-DD format",
                    },
                    "end_date": {
                        "type": "string",
                        "description": "End date in YYYY-MM-DD format",
                    },
                },
                "required": ["start_date", "end_date"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_member_insights",
            "description": (
                "Get member/player insights for the club. Shows total members, "
                "top bookers, new vs returning players, and booking frequency. "
                "Use when the user asks about members, players, or customer analytics."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "start_date": {
                        "type": "string",
                        "description": "Start date for booking analysis in YYYY-MM-DD format",
                    },
                    "end_date": {
                        "type": "string",
                        "description": "End date for booking analysis in YYYY-MM-DD format",
                    },
                },
                "required": ["start_date", "end_date"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_operational_alerts",
            "description": (
                "Get operational alerts and insights: cancellation rates, peak hours, "
                "underutilized time slots, and booking type distribution. "
                "Use when the user asks about cancellations, peak times, or operational health."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "start_date": {
                        "type": "string",
                        "description": "Start date in YYYY-MM-DD format",
                    },
                    "end_date": {
                        "type": "string",
                        "description": "End date in YYYY-MM-DD format",
                    },
                },
                "required": ["start_date", "end_date"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_available_slots",
            "description": (
                "Get available (unbooked) court slots for a specific date. "
                "Shows which courts have open time slots. "
                "Use when the user asks about availability or free slots."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "date": {
                        "type": "string",
                        "description": "The date to check in YYYY-MM-DD format",
                    },
                },
                "required": ["date"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_booking_details",
            "description": (
                "Get detailed booking information including player/participant names, "
                "court assignments, times, prices, and payment status. "
                "Can filter by court name and/or player name. "
                "Use this when the user asks WHO played on a specific court, "
                "WHO booked yesterday, which players used a court, or any question "
                "about specific bookings and participants."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "date": {
                        "type": "string",
                        "description": "The date to check in YYYY-MM-DD format",
                    },
                    "court_name": {
                        "type": "string",
                        "description": (
                            "Optional: filter by court name (partial match, case-insensitive). "
                            "E.g. 'Hirostar', 'Court 1', 'Pista 4', 'Estadio'"
                        ),
                    },
                    "player_name": {
                        "type": "string",
                        "description": (
                            "Optional: filter by player name (partial match, case-insensitive). "
                            "E.g. 'Axel', 'Molina'"
                        ),
                    },
                    "include_canceled": {
                        "type": "boolean",
                        "description": "Whether to include canceled bookings. Default false.",
                    },
                },
                "required": ["date"],
            },
        },
    },
]


# ── Tool execution functions ────────────────────────────────────────────


def _parse_price(price_str: str) -> float:
    """Parse a price string like '10 EUR' into a float."""
    if not price_str:
        return 0.0
    try:
        return float(price_str.split()[0])
    except (ValueError, IndexError):
        return 0.0


def _duration_minutes(microseconds: int) -> int:
    """Convert duration in microseconds to minutes."""
    return microseconds // 1_000_000 // 60


def _extract_participants(booking: dict) -> list[str]:
    """Extract participant names from a booking."""
    participants = booking.get("participant_info", {}).get("participants", [])
    names = []
    for p in participants:
        name = p.get("name", "").strip()
        if name:
            names.append(name)
    return names if names else ["Unknown"]


def execute_get_occupancy_for_date(
    api: PlaytomicAPI, tenant_id: str, date_str: str
) -> dict:
    """Get occupancy for a single date."""
    target = datetime.strptime(date_str, "%Y-%m-%d")

    bookings = api.get_bookings_for_date(tenant_id, target)
    availability = []
    try:
        availability = api.get_availability(tenant_id, target)
    except Exception:
        pass

    # Analyze bookings by court
    courts = {}
    for b in bookings:
        if b.get("is_canceled"):
            continue
        court = b.get("resource_name", "Unknown")
        if court not in courts:
            courts[court] = []
        start_readable = _utc_to_readable_time(b.get("booking_start_date", ""))
        end_readable = _utc_to_readable_time(b.get("booking_end_date", ""))
        # Keep ISO for charts
        start_iso = _utc_to_local(b.get("booking_start_date", ""))
        courts[court].append({
            "time": f"{start_readable} - {end_readable}",
            "start_iso": start_iso,
            "players": _extract_participants(b),
            "type": b.get("booking_type", "UNKNOWN"),
            "status": b.get("status", "UNKNOWN"),
            "payment_status": b.get("payment_status", "UNKNOWN"),
            "price": b.get("price", "0"),
        })

    # Count active (non-canceled) bookings
    active_bookings = [b for b in bookings if not b.get("is_canceled")]

    # Available slots per court
    available_slots = {}
    for resource in availability:
        resource_id = resource.get("resource_id", "Unknown")
        slots = resource.get("slots", [])
        available_slots[resource_id] = [
            {"start_time": s.get("start_time"), "duration_min": s.get("duration", 0)}
            for s in slots
        ]

    total_available = sum(len(s) for s in available_slots.values())
    total_booked = len(active_bookings)

    occupancy_pct = 0
    if total_booked + total_available > 0:
        occupancy_pct = round(total_booked / (total_booked + total_available) * 100, 1)

    return {
        "date": date_str,
        "total_bookings": total_booked,
        "total_available_slots": total_available,
        "occupancy_percentage": occupancy_pct,
        "courts": {
            court: {
                "bookings_count": len(slots),
                "time_slots": slots,
            }
            for court, slots in courts.items()
        },
        "available_slots_per_court": {
            rid: len(slots) for rid, slots in available_slots.items()
        },
    }


def execute_get_occupancy_for_range(
    api: PlaytomicAPI, tenant_id: str, start_str: str, end_str: str
) -> dict:
    """Get occupancy summary for a date range."""
    start = datetime.strptime(start_str, "%Y-%m-%d")
    end = datetime.strptime(end_str, "%Y-%m-%d")

    bookings = api.get_bookings_for_range(
        tenant_id,
        start.replace(hour=0, minute=0, second=0),
        end.replace(hour=23, minute=59, second=59),
    )

    active = [b for b in bookings if not b.get("is_canceled")]
    canceled = [b for b in bookings if b.get("is_canceled")]

    # Group by date (using local time)
    daily = {}
    for b in active:
        bdate = _utc_to_local_date(b.get("booking_start_date", ""))
        daily.setdefault(bdate, 0)
        daily[bdate] += 1

    # Group by court
    by_court = {}
    for b in active:
        court = b.get("resource_name", "Unknown")
        by_court.setdefault(court, 0)
        by_court[court] += 1

    return {
        "period": f"{start_str} to {end_str}",
        "timezone": str(_club_tz),
        "total_bookings": len(active),
        "total_canceled": len(canceled),
        "daily_breakdown": dict(sorted(daily.items())),
        "bookings_by_court": by_court,
        "busiest_day": max(daily, key=daily.get) if daily else "N/A",
        "quietest_day": min(daily, key=daily.get) if daily else "N/A",
    }


def execute_get_revenue_summary(
    api: PlaytomicAPI, tenant_id: str, start_str: str, end_str: str
) -> dict:
    """Calculate revenue metrics from bookings."""
    start = datetime.strptime(start_str, "%Y-%m-%d")
    end = datetime.strptime(end_str, "%Y-%m-%d")

    bookings = api.get_bookings_for_range(
        tenant_id,
        start.replace(hour=0, minute=0, second=0),
        end.replace(hour=23, minute=59, second=59),
    )

    active = [b for b in bookings if not b.get("is_canceled")]

    total_revenue = sum(_parse_price(b.get("price", "0")) for b in active)
    avg_value = total_revenue / len(active) if active else 0

    # Revenue by payment status
    by_payment = {}
    for b in active:
        ps = b.get("payment_status", "UNKNOWN")
        by_payment.setdefault(ps, {"count": 0, "revenue": 0})
        by_payment[ps]["count"] += 1
        by_payment[ps]["revenue"] += _parse_price(b.get("price", "0"))

    # Revenue by court
    by_court = {}
    for b in active:
        court = b.get("resource_name", "Unknown")
        by_court.setdefault(court, {"count": 0, "revenue": 0})
        by_court[court]["count"] += 1
        by_court[court]["revenue"] += _parse_price(b.get("price", "0"))

    # Revenue by date (local time)
    by_date = {}
    for b in active:
        bdate = _utc_to_local_date(b.get("booking_start_date", ""))
        by_date.setdefault(bdate, 0)
        by_date[bdate] += _parse_price(b.get("price", "0"))

    # Round all revenue figures
    for k in by_payment:
        by_payment[k]["revenue"] = round(by_payment[k]["revenue"], 2)
    for k in by_court:
        by_court[k]["revenue"] = round(by_court[k]["revenue"], 2)
    by_date = {k: round(v, 2) for k, v in sorted(by_date.items())}

    return {
        "period": f"{start_str} to {end_str}",
        "total_revenue": round(total_revenue, 2),
        "total_bookings": len(active),
        "average_booking_value": round(avg_value, 2),
        "by_payment_status": by_payment,
        "by_court": by_court,
        "daily_revenue": by_date,
        "currency": "EUR",
    }


def execute_get_member_insights(
    api: PlaytomicAPI, tenant_id: str, start_str: str, end_str: str
) -> dict:
    """Analyze member/player data combined with booking patterns."""
    start = datetime.strptime(start_str, "%Y-%m-%d")
    end = datetime.strptime(end_str, "%Y-%m-%d")

    # Fetch players
    players = []
    try:
        players = api.get_players(tenant_id)
    except Exception as e:
        players = []

    # Fetch bookings for the period
    bookings = api.get_bookings_for_range(
        tenant_id,
        start.replace(hour=0, minute=0, second=0),
        end.replace(hour=23, minute=59, second=59),
    )
    active = [b for b in bookings if not b.get("is_canceled")]

    # Count bookings per participant
    player_bookings = {}
    for b in active:
        participants = b.get("participant_info", {}).get("participants", [])
        for p in participants:
            pid = p.get("participant_id", "unknown")
            name = p.get("name", "Unknown")
            player_bookings.setdefault(pid, {"name": name, "count": 0})
            player_bookings[pid]["count"] += 1

    # Top bookers
    top_bookers = sorted(
        player_bookings.values(), key=lambda x: x["count"], reverse=True
    )[:10]

    # Player level distribution (from player data)
    levels = {}
    for p in players:
        for sport in p.get("sports", []):
            if sport.get("sport_id") == "PADEL":
                level = sport.get("level_value", 0)
                bucket = f"{level:.1f}"
                levels.setdefault(bucket, 0)
                levels[bucket] += 1

    return {
        "period": f"{start_str} to {end_str}",
        "total_registered_players": len(players),
        "active_players_in_period": len(player_bookings),
        "top_bookers": top_bookers,
        "total_bookings": len(active),
        "avg_bookings_per_active_player": (
            round(len(active) / len(player_bookings), 1) if player_bookings else 0
        ),
        "padel_level_distribution": dict(sorted(levels.items())),
    }


def execute_get_operational_alerts(
    api: PlaytomicAPI, tenant_id: str, start_str: str, end_str: str
) -> dict:
    """Generate operational insights and alerts."""
    start = datetime.strptime(start_str, "%Y-%m-%d")
    end = datetime.strptime(end_str, "%Y-%m-%d")

    bookings = api.get_bookings_for_range(
        tenant_id,
        start.replace(hour=0, minute=0, second=0),
        end.replace(hour=23, minute=59, second=59),
    )

    active = [b for b in bookings if not b.get("is_canceled")]
    canceled = [b for b in bookings if b.get("is_canceled")]

    cancellation_rate = (
        round(len(canceled) / len(bookings) * 100, 1) if bookings else 0
    )

    # Peak hours analysis (converted to local time)
    hour_counts = {}
    for b in active:
        hour = _utc_to_local_hour(b.get("booking_start_date", ""))
        hour_counts.setdefault(hour, 0)
        hour_counts[hour] += 1

    sorted_hours = sorted(hour_counts.items(), key=lambda x: x[1], reverse=True)
    peak_hours = [{"hour": f"{h:02d}:00", "bookings": c} for h, c in sorted_hours[:5]]
    quiet_hours = [{"hour": f"{h:02d}:00", "bookings": c} for h, c in sorted_hours[-5:]] if len(sorted_hours) >= 5 else []

    # Booking type distribution
    type_dist = {}
    for b in active:
        bt = b.get("booking_type", "UNKNOWN")
        type_dist.setdefault(bt, 0)
        type_dist[bt] += 1

    # Day of week distribution (using local time)
    dow_counts = {}
    day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    for b in active:
        bdate_str = _utc_to_local_date(b.get("booking_start_date", ""))
        try:
            bdate = datetime.strptime(bdate_str, "%Y-%m-%d")
            day = day_names[bdate.weekday()]
            dow_counts.setdefault(day, 0)
            dow_counts[day] += 1
        except ValueError:
            pass

    # Unpaid bookings alert
    unpaid = [
        b for b in active
        if b.get("payment_status") in ("UNPAID", "PENDING")
    ]
    unpaid_revenue = sum(_parse_price(b.get("price", "0")) for b in unpaid)

    alerts = []
    if cancellation_rate > 15:
        alerts.append(f"High cancellation rate: {cancellation_rate}% (above 15% threshold)")
    if unpaid_revenue > 0:
        alerts.append(f"Unpaid/pending revenue: {round(unpaid_revenue, 2)} EUR across {len(unpaid)} bookings")
    if quiet_hours:
        quietest = quiet_hours[0]["hour"]
        alerts.append(f"Most underutilized time slot: {quietest}")

    return {
        "period": f"{start_str} to {end_str}",
        "total_bookings": len(active),
        "total_canceled": len(canceled),
        "cancellation_rate_pct": cancellation_rate,
        "peak_hours": peak_hours,
        "quiet_hours": quiet_hours,
        "booking_type_distribution": type_dist,
        "day_of_week_distribution": dow_counts,
        "unpaid_bookings": len(unpaid),
        "unpaid_revenue_eur": round(unpaid_revenue, 2),
        "alerts": alerts,
    }


def execute_get_available_slots(
    api: PlaytomicAPI, tenant_id: str, date_str: str
) -> dict:
    """Get available court slots for a date."""
    target = datetime.strptime(date_str, "%Y-%m-%d")

    availability = api.get_availability(tenant_id, target)

    courts = {}
    total_slots = 0
    for resource in availability:
        rid = resource.get("resource_id", "Unknown")
        slots = resource.get("slots", [])
        courts[rid] = [
            {
                "start_time": s.get("start_time"),
                "duration_min": s.get("duration", 0),
                "price": s.get("price", "N/A"),
            }
            for s in slots
        ]
        total_slots += len(slots)

    return {
        "date": date_str,
        "total_available_slots": total_slots,
        "courts_with_availability": len(courts),
        "slots_by_court": courts,
    }


def execute_get_booking_details(
    api: PlaytomicAPI,
    tenant_id: str,
    date_str: str,
    court_name: Optional[str] = None,
    player_name: Optional[str] = None,
    include_canceled: bool = False,
) -> dict:
    """Get detailed booking info with participant names, filterable by court/player."""
    target = datetime.strptime(date_str, "%Y-%m-%d")
    bookings = api.get_bookings_for_date(tenant_id, target)

    results = []
    for b in bookings:
        # Filter canceled
        if b.get("is_canceled") and not include_canceled:
            continue

        resource = b.get("resource_name", "Unknown")
        players = _extract_participants(b)
        start_readable = _utc_to_readable_time(b.get("booking_start_date", ""))
        end_readable = _utc_to_readable_time(b.get("booking_end_date", ""))
        start_iso = _utc_to_local(b.get("booking_start_date", ""))

        # Filter by court name (partial, case-insensitive)
        if court_name and court_name.lower() not in resource.lower():
            continue

        # Filter by player name (partial, case-insensitive)
        if player_name:
            match = any(player_name.lower() in p.lower() for p in players)
            if not match:
                continue

        # Get participant details
        participant_details = []
        for p in b.get("participant_info", {}).get("participants", []):
            detail = {"name": p.get("name", "Unknown").strip()}
            if p.get("email"):
                detail["email"] = p["email"]
            detail["type"] = p.get("participant_type", "UNKNOWN")
            participant_details.append(detail)

        results.append({
            "court": resource,
            "time": f"{start_readable} - {end_readable}",
            "start_iso": start_iso,
            "players": players,
            "participant_details": participant_details,
            "booking_type": b.get("booking_type", "UNKNOWN"),
            "status": b.get("status", "UNKNOWN"),
            "is_canceled": b.get("is_canceled", False),
            "price": b.get("price", "0"),
            "payment_status": b.get("payment_status", "UNKNOWN"),
            "origin": b.get("origin", "UNKNOWN"),
        })

    # Sort by court then start time
    results.sort(key=lambda x: (x["court"], x["start_iso"]))

    return {
        "date": date_str,
        "filters_applied": {
            "court_name": court_name,
            "player_name": player_name,
            "include_canceled": include_canceled,
        },
        "total_bookings": len(results),
        "bookings": results,
    }


# ── Tool dispatcher ─────────────────────────────────────────────────────

TOOL_EXECUTORS = {
    "get_occupancy_for_date": lambda api, tid, args: execute_get_occupancy_for_date(
        api, tid, args["date"]
    ),
    "get_occupancy_for_range": lambda api, tid, args: execute_get_occupancy_for_range(
        api, tid, args["start_date"], args["end_date"]
    ),
    "get_revenue_summary": lambda api, tid, args: execute_get_revenue_summary(
        api, tid, args["start_date"], args["end_date"]
    ),
    "get_member_insights": lambda api, tid, args: execute_get_member_insights(
        api, tid, args["start_date"], args["end_date"]
    ),
    "get_operational_alerts": lambda api, tid, args: execute_get_operational_alerts(
        api, tid, args["start_date"], args["end_date"]
    ),
    "get_available_slots": lambda api, tid, args: execute_get_available_slots(
        api, tid, args["date"]
    ),
    "get_booking_details": lambda api, tid, args: execute_get_booking_details(
        api, tid, args["date"],
        court_name=args.get("court_name"),
        player_name=args.get("player_name"),
        include_canceled=args.get("include_canceled", False),
    ),
}


# ── System prompt ───────────────────────────────────────────────────────

def build_system_prompt(tenant_id: str, timezone_name: str = "America/Cancun") -> str:
    today = date.today()
    return f"""You are the AI assistant for a padel club manager in Cancun, Mexico.
You help answer questions about the club's operations using real-time data from Playtomic.

Today's date: {today.isoformat()} ({today.strftime('%A')})
Club tenant ID: {tenant_id}
Club timezone: {timezone_name}

IMPORTANT: All times in the data have already been converted to the club's local
timezone ({timezone_name}). Present all times as local time. Do NOT mention UTC
to the user — everything is already in local time.

Your capabilities:
- Court occupancy: Check how busy the club is on any date or range
- Booking details: See WHO played on which court, filter by court name or player name
- Revenue analytics: Revenue by day, court, payment status, and averages
- Member insights: Player activity, top bookers, level distribution
- Operational alerts: Cancellation rates, peak/quiet hours, unpaid bookings

When the user asks about specific players or who used a court, use the get_booking_details tool.
This tool supports filtering by court name (e.g. "Hirostar", "Pista 4") and player name.

When the user asks about dates:
- "tomorrow" = {(today + timedelta(days=1)).isoformat()}
- "next week" = {(today + timedelta(days=(7 - today.weekday()))).isoformat()} to {(today + timedelta(days=(13 - today.weekday()))).isoformat()}
- "this week" = {(today - timedelta(days=today.weekday())).isoformat()} to {(today + timedelta(days=(6 - today.weekday()))).isoformat()}
- "this month" = {today.replace(day=1).isoformat()} to {today.isoformat()}
- "last month": calculate the first and last day of the previous month
- For "next Thursday", "next Friday", etc., calculate the correct date

When presenting data:
- Be conversational but data-driven
- Highlight key metrics and notable patterns
- If something looks unusual (high cancellations, low occupancy), mention it proactively
- Use percentages and comparisons where helpful
- Format times in a human-readable way (e.g. "9:00 AM" instead of "09:00:00")
- Always note if data seems incomplete or unavailable
- When showing revenue, always include the currency (EUR)

Important:
- Historical data is limited to the past 90 days.
- You can only read data, not create or modify bookings.
"""


# ── Main agent class ────────────────────────────────────────────────────


class PlaytomicAgent:
    """OpenAI-powered conversational agent for Playtomic club management."""

    def __init__(
        self,
        api: PlaytomicAPI,
        tenant_id: str,
        openai_api_key: str,
        model: str = "gpt-4o",
        timezone_name: str = "America/Cancun",
    ):
        self.api = api
        self.tenant_id = tenant_id
        self.client = openai.OpenAI(api_key=openai_api_key)
        self.model = model
        set_club_timezone(timezone_name)
        self.system_prompt = build_system_prompt(tenant_id, timezone_name)
        self.messages: list[dict] = [
            {"role": "system", "content": self.system_prompt}
        ]

    def reset_conversation(self):
        """Clear conversation history, keeping the system prompt."""
        self.messages = [{"role": "system", "content": self.system_prompt}]

    def chat(self, user_message: str) -> tuple[str, list[tuple[str, dict]]]:
        """
        Send a user message and get a response.
        Handles tool calls automatically in a loop.

        Returns:
            (text_response, chart_data) where chart_data is a list of
            (tool_name, result_dict) tuples for rendering charts.
        """
        self.messages.append({"role": "user", "content": user_message})
        chart_data: list[tuple[str, dict]] = []

        max_iterations = 5
        for _ in range(max_iterations):
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.messages,
                tools=TOOLS,
                tool_choice="auto",
            )

            message = response.choices[0].message
            self.messages.append(message.model_dump())

            # If no tool calls, return the text response
            if not message.tool_calls:
                return message.content or "", chart_data

            # Execute each tool call
            for tool_call in message.tool_calls:
                fn_name = tool_call.function.name
                fn_args = json.loads(tool_call.function.arguments)

                executor = TOOL_EXECUTORS.get(fn_name)
                if executor:
                    try:
                        result = executor(self.api, self.tenant_id, fn_args)
                        result_str = json.dumps(result, indent=2, default=str)
                        chart_data.append((fn_name, result))
                    except Exception as e:
                        result_str = json.dumps({"error": str(e)})
                else:
                    result_str = json.dumps({"error": f"Unknown tool: {fn_name}"})

                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result_str,
                })

        return "I apologize, but I wasn't able to complete the analysis. Please try rephrasing your question.", chart_data
