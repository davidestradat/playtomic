"""
Agente LLM para UtopIA — Utopia Padel Cancún

Usa OpenAI GPT con function calling para responder preguntas en lenguaje
natural sobre ocupación, ingresos, jugadores y operaciones del club.
"""

import json
from datetime import datetime, timedelta, date, timezone
from typing import Optional
from zoneinfo import ZoneInfo

import openai
import pandas as pd

from playtomic_api import PlaytomicAPI

# ── Helpers de zona horaria ────────────────────────────────────────────

_club_tz: ZoneInfo = ZoneInfo("America/Cancun")


def set_club_timezone(tz_name: str):
    """Establece la zona horaria del club para todas las conversiones."""
    global _club_tz
    _club_tz = ZoneInfo(tz_name)


def _utc_to_local_dt(utc_str: str) -> Optional[datetime]:
    """Convierte una cadena UTC a un datetime local con zona horaria."""
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
    """Convierte una cadena UTC (YYYY-MM-DDTHH:MM:SS) a ISO local."""
    dt = _utc_to_local_dt(utc_str)
    return dt.strftime("%Y-%m-%dT%H:%M:%S") if dt else utc_str


def _utc_to_local_date(utc_str: str) -> str:
    """Convierte una cadena UTC a fecha local (YYYY-MM-DD)."""
    dt = _utc_to_local_dt(utc_str)
    return dt.strftime("%Y-%m-%d") if dt else utc_str[:10] if utc_str else ""


def _utc_to_readable_time(utc_str: str) -> str:
    """Convierte UTC a hora local legible como '7:00 PM'."""
    dt = _utc_to_local_dt(utc_str)
    if not dt:
        return utc_str
    return dt.strftime("%-I:%M %p")


def _utc_to_local_hour(utc_str: str) -> int:
    """Convierte UTC a hora local (0-23) para agregación."""
    dt = _utc_to_local_dt(utc_str)
    return dt.hour if dt else 0


# ── Definiciones de herramientas para OpenAI function calling ─────────

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_occupancy_for_date",
            "description": (
                "Obtiene la ocupación de canchas y detalles de reservas para una fecha específica. "
                "Muestra qué tan ocupado está el club, qué canchas están reservadas y horarios disponibles. "
                "Usar cuando el usuario pregunta qué tan lleno está el club en un día específico."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "date": {
                        "type": "string",
                        "description": "La fecha a consultar en formato YYYY-MM-DD",
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
                "Obtiene un resumen de ocupación de canchas para un rango de días. "
                "Usar cuando el usuario pregunta sobre una semana, fin de semana o rango de fechas."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "start_date": {
                        "type": "string",
                        "description": "Fecha de inicio en formato YYYY-MM-DD",
                    },
                    "end_date": {
                        "type": "string",
                        "description": "Fecha de fin en formato YYYY-MM-DD",
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
                "Obtiene datos de ingresos para un rango de fechas. Muestra ingresos totales, "
                "reservas por estado de pago, valor promedio de reserva e ingresos por cancha. "
                "Usar cuando el usuario pregunta sobre ingresos, facturación o rendimiento financiero."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "start_date": {
                        "type": "string",
                        "description": "Fecha de inicio en formato YYYY-MM-DD",
                    },
                    "end_date": {
                        "type": "string",
                        "description": "Fecha de fin en formato YYYY-MM-DD",
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
                "Obtiene información sobre miembros/jugadores del club. Muestra total de miembros, "
                "jugadores que más reservan, nuevos vs recurrentes y frecuencia de reservas. "
                "Usar cuando el usuario pregunta sobre miembros, jugadores o analítica de clientes."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "start_date": {
                        "type": "string",
                        "description": "Fecha de inicio para el análisis en formato YYYY-MM-DD",
                    },
                    "end_date": {
                        "type": "string",
                        "description": "Fecha de fin para el análisis en formato YYYY-MM-DD",
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
                "Obtiene alertas operativas: tasas de cancelación, horas pico, "
                "horarios subutilizados y distribución de tipos de reserva. "
                "Usar cuando el usuario pregunta sobre cancelaciones, horarios pico o salud operativa."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "start_date": {
                        "type": "string",
                        "description": "Fecha de inicio en formato YYYY-MM-DD",
                    },
                    "end_date": {
                        "type": "string",
                        "description": "Fecha de fin en formato YYYY-MM-DD",
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
                "Obtiene los horarios disponibles (sin reservar) para una fecha específica. "
                "Muestra qué canchas tienen horarios libres. "
                "Usar cuando el usuario pregunta sobre disponibilidad o horarios libres."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "date": {
                        "type": "string",
                        "description": "La fecha a consultar en formato YYYY-MM-DD",
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
                "Obtiene información detallada de reservas incluyendo nombres de jugadores/participantes, "
                "asignación de canchas, horarios, precios y estado de pago. "
                "Puede filtrar por nombre de cancha y/o nombre de jugador. "
                "Usar cuando el usuario pregunta QUIÉN jugó en una cancha específica, "
                "QUIÉN reservó ayer, qué jugadores usaron una cancha, o cualquier pregunta "
                "sobre reservas específicas y participantes."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "date": {
                        "type": "string",
                        "description": "La fecha a consultar en formato YYYY-MM-DD",
                    },
                    "court_name": {
                        "type": "string",
                        "description": (
                            "Opcional: filtrar por nombre de cancha (coincidencia parcial, no distingue mayúsculas). "
                            "Ej: 'Hirostar', 'Cancha 1', 'Pista 4', 'Estadio'"
                        ),
                    },
                    "player_name": {
                        "type": "string",
                        "description": (
                            "Opcional: filtrar por nombre de jugador (coincidencia parcial, no distingue mayúsculas). "
                            "Ej: 'Axel', 'Molina'"
                        ),
                    },
                    "include_canceled": {
                        "type": "boolean",
                        "description": "Si se deben incluir reservas canceladas. Por defecto false.",
                    },
                },
                "required": ["date"],
            },
        },
    },
]


# ── Funciones de ejecución de herramientas ─────────────────────────────


def _parse_price(price_str: str) -> float:
    """Parsea un string de precio como '10 EUR' a float."""
    if not price_str:
        return 0.0
    try:
        return float(price_str.split()[0])
    except (ValueError, IndexError):
        return 0.0


def _duration_minutes(microseconds: int) -> int:
    """Convierte duración en microsegundos a minutos."""
    return microseconds // 1_000_000 // 60


def _extract_participants(booking: dict) -> list[str]:
    """Extrae nombres de participantes de una reserva."""
    participants = booking.get("participant_info", {}).get("participants", [])
    names = []
    for p in participants:
        name = p.get("name", "").strip()
        if name:
            names.append(name)
    return names if names else ["Desconocido"]


def execute_get_occupancy_for_date(
    api: PlaytomicAPI, tenant_id: str, date_str: str
) -> dict:
    """Obtiene ocupación para una fecha específica."""
    target = datetime.strptime(date_str, "%Y-%m-%d")

    bookings = api.get_bookings_for_date(tenant_id, target)
    availability = []
    try:
        availability = api.get_availability(tenant_id, target)
    except Exception:
        pass

    # Analizar reservas por cancha
    courts = {}
    for b in bookings:
        if b.get("is_canceled"):
            continue
        court = b.get("resource_name", "Desconocido")
        if court not in courts:
            courts[court] = []
        start_readable = _utc_to_readable_time(b.get("booking_start_date", ""))
        end_readable = _utc_to_readable_time(b.get("booking_end_date", ""))
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

    active_bookings = [b for b in bookings if not b.get("is_canceled")]

    available_slots = {}
    for resource in availability:
        resource_id = resource.get("resource_id", "Desconocido")
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
    """Obtiene resumen de ocupación para un rango de fechas."""
    start = datetime.strptime(start_str, "%Y-%m-%d")
    end = datetime.strptime(end_str, "%Y-%m-%d")

    bookings = api.get_bookings_for_range(
        tenant_id,
        start.replace(hour=0, minute=0, second=0),
        end.replace(hour=23, minute=59, second=59),
    )

    active = [b for b in bookings if not b.get("is_canceled")]
    canceled = [b for b in bookings if b.get("is_canceled")]

    daily = {}
    for b in active:
        bdate = _utc_to_local_date(b.get("booking_start_date", ""))
        daily.setdefault(bdate, 0)
        daily[bdate] += 1

    by_court = {}
    for b in active:
        court = b.get("resource_name", "Desconocido")
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
    """Calcula métricas de ingresos a partir de reservas."""
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

    by_payment = {}
    for b in active:
        ps = b.get("payment_status", "UNKNOWN")
        by_payment.setdefault(ps, {"count": 0, "revenue": 0})
        by_payment[ps]["count"] += 1
        by_payment[ps]["revenue"] += _parse_price(b.get("price", "0"))

    by_court = {}
    for b in active:
        court = b.get("resource_name", "Desconocido")
        by_court.setdefault(court, {"count": 0, "revenue": 0})
        by_court[court]["count"] += 1
        by_court[court]["revenue"] += _parse_price(b.get("price", "0"))

    by_date = {}
    for b in active:
        bdate = _utc_to_local_date(b.get("booking_start_date", ""))
        by_date.setdefault(bdate, 0)
        by_date[bdate] += _parse_price(b.get("price", "0"))

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
    """Analiza datos de miembros/jugadores con patrones de reserva."""
    start = datetime.strptime(start_str, "%Y-%m-%d")
    end = datetime.strptime(end_str, "%Y-%m-%d")

    players = []
    try:
        players = api.get_players(tenant_id)
    except Exception as e:
        players = []

    bookings = api.get_bookings_for_range(
        tenant_id,
        start.replace(hour=0, minute=0, second=0),
        end.replace(hour=23, minute=59, second=59),
    )
    active = [b for b in bookings if not b.get("is_canceled")]

    player_bookings = {}
    for b in active:
        participants = b.get("participant_info", {}).get("participants", [])
        for p in participants:
            pid = p.get("participant_id", "unknown")
            name = p.get("name", "Desconocido")
            player_bookings.setdefault(pid, {"name": name, "count": 0})
            player_bookings[pid]["count"] += 1

    top_bookers = sorted(
        player_bookings.values(), key=lambda x: x["count"], reverse=True
    )[:10]

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
    """Genera alertas e insights operativos."""
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

    hour_counts = {}
    for b in active:
        hour = _utc_to_local_hour(b.get("booking_start_date", ""))
        hour_counts.setdefault(hour, 0)
        hour_counts[hour] += 1

    sorted_hours = sorted(hour_counts.items(), key=lambda x: x[1], reverse=True)
    peak_hours = [{"hour": f"{h:02d}:00", "bookings": c} for h, c in sorted_hours[:5]]
    quiet_hours = [{"hour": f"{h:02d}:00", "bookings": c} for h, c in sorted_hours[-5:]] if len(sorted_hours) >= 5 else []

    type_dist = {}
    for b in active:
        bt = b.get("booking_type", "UNKNOWN")
        type_dist.setdefault(bt, 0)
        type_dist[bt] += 1

    dow_counts = {}
    day_names = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]
    for b in active:
        bdate_str = _utc_to_local_date(b.get("booking_start_date", ""))
        try:
            bdate = datetime.strptime(bdate_str, "%Y-%m-%d")
            day = day_names[bdate.weekday()]
            dow_counts.setdefault(day, 0)
            dow_counts[day] += 1
        except ValueError:
            pass

    unpaid = [
        b for b in active
        if b.get("payment_status") in ("UNPAID", "PENDING")
    ]
    unpaid_revenue = sum(_parse_price(b.get("price", "0")) for b in unpaid)

    alerts = []
    if cancellation_rate > 15:
        alerts.append(f"Alta tasa de cancelación: {cancellation_rate}% (supera el umbral del 15%)")
    if unpaid_revenue > 0:
        alerts.append(f"Ingresos sin cobrar/pendientes: {round(unpaid_revenue, 2)} EUR en {len(unpaid)} reservas")
    if quiet_hours:
        quietest = quiet_hours[0]["hour"]
        alerts.append(f"Horario más subutilizado: {quietest}")

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
    """Obtiene horarios disponibles para una fecha."""
    target = datetime.strptime(date_str, "%Y-%m-%d")

    availability = api.get_availability(tenant_id, target)

    courts = {}
    total_slots = 0
    for resource in availability:
        rid = resource.get("resource_id", "Desconocido")
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
    """Obtiene información detallada de reservas con nombres de participantes."""
    target = datetime.strptime(date_str, "%Y-%m-%d")
    bookings = api.get_bookings_for_date(tenant_id, target)

    results = []
    for b in bookings:
        if b.get("is_canceled") and not include_canceled:
            continue

        resource = b.get("resource_name", "Desconocido")
        players = _extract_participants(b)
        start_readable = _utc_to_readable_time(b.get("booking_start_date", ""))
        end_readable = _utc_to_readable_time(b.get("booking_end_date", ""))
        start_iso = _utc_to_local(b.get("booking_start_date", ""))

        if court_name and court_name.lower() not in resource.lower():
            continue

        if player_name:
            match = any(player_name.lower() in p.lower() for p in players)
            if not match:
                continue

        participant_details = []
        for p in b.get("participant_info", {}).get("participants", []):
            detail = {"name": p.get("name", "Desconocido").strip()}
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


# ── Despachador de herramientas ────────────────────────────────────────

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


# ── Prompt del sistema ─────────────────────────────────────────────────

def build_system_prompt(tenant_id: str, timezone_name: str = "America/Cancun") -> str:
    today = date.today()
    return f"""Eres UtopIA, el asistente inteligente de Utopia Padel Cancún.
Ayudas al administrador del club a responder preguntas sobre las operaciones usando datos en tiempo real de Playtomic.
SIEMPRE responde en español.

Fecha de hoy: {today.isoformat()} ({today.strftime('%A')})
Tenant ID del club: {tenant_id}
Zona horaria del club: {timezone_name}

IMPORTANTE: Todos los horarios en los datos ya han sido convertidos a la zona horaria local
del club ({timezone_name}). Presenta todos los horarios como hora local. NO menciones UTC
al usuario — todo ya está en hora local.

Tus capacidades:
- Ocupación de canchas: Consultar qué tan ocupado está el club en cualquier fecha o rango
- Detalles de reservas: Ver QUIÉN jugó en qué cancha, filtrar por nombre de cancha o jugador
- Analítica de ingresos: Ingresos por día, cancha, estado de pago y promedios
- Información de jugadores: Actividad de jugadores, quiénes más reservan, distribución de niveles
- Alertas operativas: Tasas de cancelación, horas pico/valle, reservas sin pagar

Cuando el usuario pregunte sobre jugadores específicos o quién usó una cancha, usa la herramienta get_booking_details.
Esta herramienta soporta filtros por nombre de cancha (ej: "Hirostar", "Pista 4") y nombre de jugador.

Cuando el usuario pregunte sobre fechas:
- "mañana" = {(today + timedelta(days=1)).isoformat()}
- "la próxima semana" = {(today + timedelta(days=(7 - today.weekday()))).isoformat()} a {(today + timedelta(days=(13 - today.weekday()))).isoformat()}
- "esta semana" = {(today - timedelta(days=today.weekday())).isoformat()} a {(today + timedelta(days=(6 - today.weekday()))).isoformat()}
- "este mes" = {today.replace(day=1).isoformat()} a {today.isoformat()}
- "el mes pasado": calcula el primer y último día del mes anterior
- Para "el próximo jueves", "el viernes que viene", etc., calcula la fecha correcta

VERIFICACIÓN DE DATOS (MUY IMPORTANTE):
Antes de responder, SIEMPRE verifica internamente:
- Cuenta manualmente las reservas en los datos — no inventes números ni redondees
- Verifica que los horarios que mencionas coincidan EXACTAMENTE con el campo "time" de cada reserva
- Cruza los nombres de jugadores con lo que aparece en los datos — no asumas ni confundas nombres
- Si los datos muestran 5 reservas, di 5 — no digas "alrededor de" ni aproximes
- Suma los ingresos uno por uno a partir de los precios individuales antes de dar un total
- Si un dato no aparece en la respuesta de la herramienta, di "no tengo esa información" en vez de inventar
- Cuando listes reservas por cancha, verifica que cada reserva esté asignada a la cancha correcta
- Si el usuario pregunta por un jugador específico, revisa TODAS las reservas para no omitir ninguna aparición

Al presentar datos:
- Sé conversacional pero preciso — el manager confía en estos datos para tomar decisiones
- Destaca métricas clave y patrones notables
- Si algo se ve inusual (muchas cancelaciones, baja ocupación), menciónalo proactivamente
- Usa porcentajes y comparaciones cuando sea útil
- Formatea los horarios de forma legible (ej: "9:00 AM" en vez de "09:00:00")
- Siempre menciona si los datos parecen incompletos o no disponibles
- Al mostrar ingresos, siempre incluye la moneda (EUR)
- Cuando presentes listas de reservas, organízalas por cancha y horario para facilitar lectura
- Si hay discrepancias en los datos (ej: totales que no cuadran), señálalas al manager

Importante:
- Los datos históricos están limitados a los últimos 90 días.
- Solo puedes leer datos, no crear ni modificar reservas.
- Tu prioridad es la EXACTITUD. Es preferible decir "no tengo esa información" a dar un dato incorrecto.
"""


# ── Clase principal del agente ─────────────────────────────────────────


class PlaytomicAgent:
    """Agente conversacional UtopIA para gestión de Utopia Padel Cancún."""

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
        """Limpia el historial de conversación, manteniendo el prompt del sistema."""
        self.messages = [{"role": "system", "content": self.system_prompt}]

    def chat(self, user_message: str) -> tuple[str, list[tuple[str, dict]]]:
        """
        Envía un mensaje del usuario y obtiene una respuesta.
        Maneja llamadas a herramientas automáticamente en un bucle.

        Retorna:
            (respuesta_texto, datos_gráficos) donde datos_gráficos es una lista de
            tuplas (nombre_herramienta, resultado_dict) para renderizar gráficos.
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

            if not message.tool_calls:
                return message.content or "", chart_data

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
                    result_str = json.dumps({"error": f"Herramienta desconocida: {fn_name}"})

                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result_str,
                })

        return "Disculpa, no pude completar el análisis. Por favor intenta reformular tu pregunta.", chart_data
