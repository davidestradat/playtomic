"""
Constructores de gráficos Plotly para Playtomic Club Manager.

Cada función recibe datos crudos de las herramientas y retorna una lista de
figuras Plotly listas para renderizar en Streamlit.
"""

import plotly.graph_objects as go
import plotly.express as px

# ── Tema compartido ────────────────────────────────────────────────────

COLORS = {
    "primary": "#1B998B",
    "secondary": "#2D3047",
    "accent": "#E84855",
    "warning": "#FF9B71",
    "success": "#1B998B",
    "muted": "#8D99AE",
    "bg": "#FFFFFF",
}

PALETTE = ["#1B998B", "#2D3047", "#E84855", "#FF9B71", "#5C6BC0", "#AB47BC", "#26A69A", "#EF5350"]

LAYOUT_BASE = dict(
    font=dict(family="Inter, sans-serif", size=13),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=40, r=20, t=50, b=40),
)


def _apply_layout(fig: go.Figure, title: str, height: int = 380) -> go.Figure:
    fig.update_layout(title=dict(text=title, font_size=16), height=height, **LAYOUT_BASE)
    fig.update_xaxes(showgrid=True, gridcolor="#f0f0f0")
    fig.update_yaxes(showgrid=True, gridcolor="#f0f0f0")
    return fig


# ── Ocupación: fecha específica ────────────────────────────────────────

def chart_occupancy_date(data: dict) -> list[go.Figure]:
    figures = []

    # 1. Reservas por cancha — barras horizontales
    courts = data.get("courts", {})
    if courts:
        names = list(courts.keys())
        counts = [c["bookings_count"] for c in courts.values()]

        fig = go.Figure(go.Bar(
            x=counts, y=names, orientation="h",
            marker_color=COLORS["primary"],
            text=counts, textposition="auto",
        ))
        _apply_layout(fig, f"Reservas por Cancha — {data['date']}")
        fig.update_xaxes(title_text="Reservas")
        figures.append(fig)

    # 2. Indicador de ocupación
    pct = data.get("occupancy_percentage", 0)
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=pct,
        number={"suffix": "%"},
        title={"text": f"Ocupación de Canchas — {data['date']}"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": COLORS["primary"]},
            "steps": [
                {"range": [0, 30], "color": "#e8f5e9"},
                {"range": [30, 70], "color": "#fff9c4"},
                {"range": [70, 100], "color": "#ffebee"},
            ],
            "threshold": {
                "line": {"color": COLORS["accent"], "width": 3},
                "thickness": 0.8,
                "value": 85,
            },
        },
    ))
    _apply_layout(fig, f"Ocupación de Canchas — {data['date']}", height=300)
    figures.append(fig)

    # 3. Mapa de calor por cancha y hora
    if courts:
        court_names = []
        hours = []
        for court, info in courts.items():
            for slot in info.get("time_slots", []):
                start_iso = slot.get("start_iso", "")
                if "T" in start_iso:
                    h = int(start_iso.split("T")[1][:2])
                    court_names.append(court)
                    hours.append(h)

        if court_names:
            unique_courts = sorted(set(court_names))
            all_hours = list(range(7, 24))
            matrix = []
            for c in unique_courts:
                row = []
                for h in all_hours:
                    count = sum(1 for cn, hr in zip(court_names, hours) if cn == c and hr == h)
                    row.append(count)
                matrix.append(row)

            fig = go.Figure(go.Heatmap(
                z=matrix,
                x=[f"{h:02d}:00" for h in all_hours],
                y=unique_courts,
                colorscale=[[0, "#f5f5f5"], [0.5, "#80cbc4"], [1, "#00695c"]],
                showscale=True,
                colorbar_title="Reservas",
            ))
            _apply_layout(fig, f"Mapa de Uso por Cancha — {data['date']}")
            fig.update_xaxes(title_text="Hora")
            figures.append(fig)

    return figures


# ── Ocupación: rango de fechas ─────────────────────────────────────────

def chart_occupancy_range(data: dict) -> list[go.Figure]:
    figures = []

    # 1. Reservas diarias — línea
    daily = data.get("daily_breakdown", {})
    if daily:
        dates = list(daily.keys())
        counts = list(daily.values())

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dates, y=counts, mode="lines+markers",
            line=dict(color=COLORS["primary"], width=3),
            marker=dict(size=8),
            fill="tozeroy",
            fillcolor="rgba(27,153,139,0.15)",
        ))
        _apply_layout(fig, f"Reservas Diarias — {data['period']}")
        fig.update_xaxes(title_text="Fecha")
        fig.update_yaxes(title_text="Reservas")
        figures.append(fig)

    # 2. Reservas por cancha — barras
    by_court = data.get("bookings_by_court", {})
    if by_court:
        courts = list(by_court.keys())
        vals = list(by_court.values())

        fig = go.Figure(go.Bar(
            x=courts, y=vals,
            marker_color=PALETTE[:len(courts)],
            text=vals, textposition="auto",
        ))
        _apply_layout(fig, f"Reservas por Cancha — {data['period']}")
        fig.update_yaxes(title_text="Reservas")
        figures.append(fig)

    return figures


# ── Ingresos ───────────────────────────────────────────────────────────

def chart_revenue(data: dict) -> list[go.Figure]:
    figures = []
    currency = data.get("currency", "EUR")

    # 1. Ingresos diarios — barras
    daily = data.get("daily_revenue", {})
    if daily:
        dates = list(daily.keys())
        vals = list(daily.values())

        fig = go.Figure(go.Bar(
            x=dates, y=vals,
            marker_color=COLORS["primary"],
            text=[f"{v:.0f}" for v in vals],
            textposition="auto",
        ))
        _apply_layout(fig, f"Ingresos Diarios ({currency}) — {data['period']}")
        fig.update_xaxes(title_text="Fecha")
        fig.update_yaxes(title_text=f"Ingresos ({currency})")
        figures.append(fig)

    # 2. Ingresos por cancha — barras + línea
    by_court = data.get("by_court", {})
    if by_court:
        courts = list(by_court.keys())
        revenues = [v["revenue"] for v in by_court.values()]
        booking_counts = [v["count"] for v in by_court.values()]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=courts, y=revenues, name="Ingresos",
            marker_color=COLORS["primary"],
            text=[f"{r:.0f}" for r in revenues],
            textposition="auto",
        ))
        fig.add_trace(go.Scatter(
            x=courts, y=booking_counts, name="Reservas",
            mode="lines+markers",
            line=dict(color=COLORS["accent"], width=2),
            marker=dict(size=8),
            yaxis="y2",
        ))
        _apply_layout(fig, f"Ingresos y Reservas por Cancha — {data['period']}")
        fig.update_layout(
            yaxis=dict(title=f"Ingresos ({currency})"),
            yaxis2=dict(title="Reservas", overlaying="y", side="right"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        figures.append(fig)

    # 3. Distribución por estado de pago — dona
    by_payment = data.get("by_payment_status", {})
    if by_payment:
        labels = list(by_payment.keys())
        values = [v["revenue"] for v in by_payment.values()]

        fig = go.Figure(go.Pie(
            labels=labels, values=values,
            hole=0.5,
            marker_colors=PALETTE[:len(labels)],
            textinfo="label+percent",
            textposition="outside",
        ))
        _apply_layout(fig, f"Ingresos por Estado de Pago — {data['period']}", height=350)
        figures.append(fig)

    return figures


# ── Información de jugadores ───────────────────────────────────────────

def chart_members(data: dict) -> list[go.Figure]:
    figures = []

    # 1. Jugadores que más reservan — barras horizontales
    top = data.get("top_bookers", [])
    if top:
        names = [b["name"] for b in reversed(top)]
        counts = [b["count"] for b in reversed(top)]

        fig = go.Figure(go.Bar(
            x=counts, y=names, orientation="h",
            marker_color=COLORS["primary"],
            text=counts, textposition="auto",
        ))
        _apply_layout(fig, f"Jugadores con Más Reservas — {data['period']}", height=max(300, len(top) * 40 + 100))
        fig.update_xaxes(title_text="Reservas")
        figures.append(fig)

    # 2. Distribución de niveles — barras
    levels = data.get("padel_level_distribution", {})
    if levels:
        lvl_labels = list(levels.keys())
        lvl_counts = list(levels.values())

        fig = go.Figure(go.Bar(
            x=lvl_labels, y=lvl_counts,
            marker_color=COLORS["secondary"],
            text=lvl_counts, textposition="auto",
        ))
        _apply_layout(fig, "Distribución de Niveles de Pádel")
        fig.update_xaxes(title_text="Nivel")
        fig.update_yaxes(title_text="Jugadores")
        figures.append(fig)

    # 3. Jugadores activos vs registrados — indicador
    total = data.get("total_registered_players", 0)
    active = data.get("active_players_in_period", 0)
    if total > 0:
        pct = round(active / total * 100, 1)
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=pct,
            number={"suffix": "%"},
            title={"text": f"Jugadores Activos ({active}/{total})"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": COLORS["primary"]},
                "steps": [
                    {"range": [0, 25], "color": "#ffebee"},
                    {"range": [25, 60], "color": "#fff9c4"},
                    {"range": [60, 100], "color": "#e8f5e9"},
                ],
            },
        ))
        _apply_layout(fig, f"Jugadores Activos ({active}/{total})", height=280)
        figures.append(fig)

    return figures


# ── Alertas operativas ─────────────────────────────────────────────────

def chart_operations(data: dict) -> list[go.Figure]:
    figures = []

    # 1. Reservas por hora — barras
    peak = data.get("peak_hours", [])
    quiet = data.get("quiet_hours", [])
    all_hours_data = {}
    for h in peak + quiet:
        all_hours_data[h["hour"]] = h["bookings"]

    if all_hours_data:
        sorted_h = sorted(all_hours_data.items(), key=lambda x: x[0])
        hours = [h[0] for h in sorted_h]
        counts = [h[1] for h in sorted_h]

        peak_set = {h["hour"] for h in peak[:3]}
        quiet_set = {h["hour"] for h in quiet[:2]} if quiet else set()
        colors = [
            COLORS["primary"] if h in peak_set
            else COLORS["accent"] if h in quiet_set
            else COLORS["muted"]
            for h in hours
        ]

        fig = go.Figure(go.Bar(
            x=hours, y=counts,
            marker_color=colors,
            text=counts, textposition="auto",
        ))
        _apply_layout(fig, f"Reservas por Hora — {data['period']}")
        fig.update_xaxes(title_text="Hora (hora local)")
        fig.update_yaxes(title_text="Reservas")
        figures.append(fig)

    # 2. Distribución por día de la semana
    dow = data.get("day_of_week_distribution", {})
    if dow:
        day_order = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]
        ordered = [(d, dow.get(d, 0)) for d in day_order if d in dow]
        days = [d[0][:3] for d in ordered]
        vals = [d[1] for d in ordered]

        fig = go.Figure(go.Bar(
            x=days, y=vals,
            marker_color=[COLORS["secondary"] if d[0] in ("Sábado", "Domingo") else COLORS["primary"] for d in ordered],
            text=vals, textposition="auto",
        ))
        _apply_layout(fig, f"Reservas por Día de la Semana — {data['period']}")
        fig.update_yaxes(title_text="Reservas")
        figures.append(fig)

    # 3. Tipos de reserva — dona
    types = data.get("booking_type_distribution", {})
    if types:
        labels = [t.replace("_", " ").title() for t in types.keys()]
        values = list(types.values())

        fig = go.Figure(go.Pie(
            labels=labels, values=values,
            hole=0.45,
            marker_colors=PALETTE[:len(labels)],
            textinfo="label+percent",
            textposition="outside",
        ))
        _apply_layout(fig, f"Tipos de Reserva — {data['period']}", height=350)
        figures.append(fig)

    # 4. Indicador de tasa de cancelación
    cancel_rate = data.get("cancellation_rate_pct", 0)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=cancel_rate,
        number={"suffix": "%"},
        title={"text": "Tasa de Cancelación"},
        gauge={
            "axis": {"range": [0, 50]},
            "bar": {"color": COLORS["accent"] if cancel_rate > 15 else COLORS["primary"]},
            "steps": [
                {"range": [0, 10], "color": "#e8f5e9"},
                {"range": [10, 20], "color": "#fff9c4"},
                {"range": [20, 50], "color": "#ffebee"},
            ],
            "threshold": {
                "line": {"color": COLORS["accent"], "width": 3},
                "thickness": 0.8,
                "value": 15,
            },
        },
    ))
    _apply_layout(fig, "Tasa de Cancelación", height=280)
    figures.append(fig)

    return figures


# ── Horarios disponibles ──────────────────────────────────────────────

def chart_available_slots(data: dict) -> list[go.Figure]:
    figures = []

    slots_by_court = data.get("slots_by_court", {})
    if not slots_by_court:
        return figures

    # 1. Horarios disponibles por cancha
    courts = list(slots_by_court.keys())
    slot_counts = [len(s) for s in slots_by_court.values()]

    fig = go.Figure(go.Bar(
        x=courts, y=slot_counts,
        marker_color=COLORS["success"],
        text=slot_counts, textposition="auto",
    ))
    _apply_layout(fig, f"Horarios Disponibles por Cancha — {data['date']}")
    fig.update_yaxes(title_text="Horarios Libres")
    figures.append(fig)

    # 2. Mapa de disponibilidad
    all_hours = list(range(7, 24))
    court_ids = sorted(slots_by_court.keys())
    matrix = []
    for cid in court_ids:
        row = [0] * len(all_hours)
        for s in slots_by_court[cid]:
            st = s.get("start_time", "")
            if st:
                try:
                    h = int(st.split(":")[0])
                    if h in all_hours:
                        row[all_hours.index(h)] = 1
                except (ValueError, IndexError):
                    pass
        matrix.append(row)

    if matrix:
        fig = go.Figure(go.Heatmap(
            z=matrix,
            x=[f"{h:02d}:00" for h in all_hours],
            y=[f"Cancha {i+1}" for i in range(len(court_ids))],
            colorscale=[[0, "#ffebee"], [1, "#c8e6c9"]],
            showscale=False,
            zmin=0, zmax=1,
        ))
        _apply_layout(fig, f"Mapa de Disponibilidad — {data['date']}")
        fig.update_xaxes(title_text="Hora")
        fig.add_annotation(
            x=1.0, y=-0.15, xref="paper", yref="paper",
            text="Verde = Disponible | Rojo = Reservado",
            showarrow=False, font_size=11, font_color=COLORS["muted"],
        )
        figures.append(fig)

    return figures


# ── Detalles de reservas ──────────────────────────────────────────────

def chart_booking_details(data: dict) -> list[go.Figure]:
    figures = []
    bookings = data.get("bookings", [])
    if not bookings:
        return figures

    courts = {}
    for b in bookings:
        court = b.get("court", "Desconocido")
        courts.setdefault(court, [])
        start_iso = b.get("start_iso", "")
        players = ", ".join(b.get("players", ["Desconocido"]))
        if "T" in start_iso:
            h = int(start_iso.split("T")[1][:2])
            m = int(start_iso.split("T")[1][3:5])
            courts[court].append({"hour": h + m / 60, "players": players, "type": b.get("booking_type", "")})

    if courts:
        all_hours = list(range(7, 24))
        court_names = sorted(courts.keys())

        matrix = []
        hover_text = []
        for c in court_names:
            row = [0] * len(all_hours)
            text_row = [""] * len(all_hours)
            for entry in courts[c]:
                h = int(entry["hour"])
                if h in all_hours:
                    idx = all_hours.index(h)
                    row[idx] = 1
                    text_row[idx] = entry["players"]
            matrix.append(row)
            hover_text.append(text_row)

        fig = go.Figure(go.Heatmap(
            z=matrix,
            x=[f"{h:02d}:00" for h in all_hours],
            y=court_names,
            customdata=hover_text,
            hovertemplate="<b>%{y}</b> a las %{x}<br>Jugadores: %{customdata}<extra></extra>",
            colorscale=[[0, "#f5f5f5"], [1, "#1B998B"]],
            showscale=False,
        ))
        filters = data.get("filters_applied", {})
        title_parts = [f"Agenda de Reservas — {data['date']}"]
        if filters.get("court_name"):
            title_parts.append(f"(Cancha: {filters['court_name']})")
        if filters.get("player_name"):
            title_parts.append(f"(Jugador: {filters['player_name']})")
        _apply_layout(fig, " ".join(title_parts))
        fig.update_xaxes(title_text="Hora (hora local)")
        figures.append(fig)

    return figures


CHART_BUILDERS = {
    "get_occupancy_for_date": chart_occupancy_date,
    "get_occupancy_for_range": chart_occupancy_range,
    "get_revenue_summary": chart_revenue,
    "get_member_insights": chart_members,
    "get_operational_alerts": chart_operations,
    "get_available_slots": chart_available_slots,
    "get_booking_details": chart_booking_details,
}


def build_charts(tool_name: str, tool_result: dict) -> list[go.Figure]:
    """Construye gráficos para un resultado de herramienta. Retorna lista vacía si no aplica."""
    builder = CHART_BUILDERS.get(tool_name)
    if builder:
        try:
            return builder(tool_result)
        except Exception:
            return []
    return []
