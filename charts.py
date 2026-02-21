"""
Plotly chart builders for Playtomic Club Manager.

Each function takes raw tool result data and returns a list of
Plotly figure objects ready to render in Streamlit.
"""

import plotly.graph_objects as go
import plotly.express as px

# ── Shared theme ────────────────────────────────────────────────────────

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


# ── Occupancy: single date ──────────────────────────────────────────────

def chart_occupancy_date(data: dict) -> list[go.Figure]:
    figures = []

    # 1. Bookings per court — horizontal bar
    courts = data.get("courts", {})
    if courts:
        names = list(courts.keys())
        counts = [c["bookings_count"] for c in courts.values()]

        fig = go.Figure(go.Bar(
            x=counts, y=names, orientation="h",
            marker_color=COLORS["primary"],
            text=counts, textposition="auto",
        ))
        _apply_layout(fig, f"Bookings per Court — {data['date']}")
        fig.update_xaxes(title_text="Bookings")
        figures.append(fig)

    # 2. Occupancy gauge
    pct = data.get("occupancy_percentage", 0)
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=pct,
        number={"suffix": "%"},
        title={"text": f"Court Occupancy — {data['date']}"},
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
    _apply_layout(fig, f"Court Occupancy — {data['date']}", height=300)
    figures.append(fig)

    # 3. Court timeline heatmap
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
            # Build a matrix: courts x hours
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
                colorbar_title="Bookings",
            ))
            _apply_layout(fig, f"Court Usage Heatmap — {data['date']}")
            fig.update_xaxes(title_text="Hour")
            figures.append(fig)

    return figures


# ── Occupancy: date range ───────────────────────────────────────────────

def chart_occupancy_range(data: dict) -> list[go.Figure]:
    figures = []

    # 1. Daily bookings line chart
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
        _apply_layout(fig, f"Daily Bookings — {data['period']}")
        fig.update_xaxes(title_text="Date")
        fig.update_yaxes(title_text="Bookings")
        figures.append(fig)

    # 2. Bookings by court — bar
    by_court = data.get("bookings_by_court", {})
    if by_court:
        courts = list(by_court.keys())
        vals = list(by_court.values())

        fig = go.Figure(go.Bar(
            x=courts, y=vals,
            marker_color=PALETTE[:len(courts)],
            text=vals, textposition="auto",
        ))
        _apply_layout(fig, f"Bookings by Court — {data['period']}")
        fig.update_yaxes(title_text="Bookings")
        figures.append(fig)

    return figures


# ── Revenue ─────────────────────────────────────────────────────────────

def chart_revenue(data: dict) -> list[go.Figure]:
    figures = []
    currency = data.get("currency", "EUR")

    # 1. Daily revenue bar chart
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
        _apply_layout(fig, f"Daily Revenue ({currency}) — {data['period']}")
        fig.update_xaxes(title_text="Date")
        fig.update_yaxes(title_text=f"Revenue ({currency})")
        figures.append(fig)

    # 2. Revenue by court — bar
    by_court = data.get("by_court", {})
    if by_court:
        courts = list(by_court.keys())
        revenues = [v["revenue"] for v in by_court.values()]
        booking_counts = [v["count"] for v in by_court.values()]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=courts, y=revenues, name="Revenue",
            marker_color=COLORS["primary"],
            text=[f"{r:.0f}" for r in revenues],
            textposition="auto",
        ))
        fig.add_trace(go.Scatter(
            x=courts, y=booking_counts, name="Bookings",
            mode="lines+markers",
            line=dict(color=COLORS["accent"], width=2),
            marker=dict(size=8),
            yaxis="y2",
        ))
        _apply_layout(fig, f"Revenue & Bookings by Court — {data['period']}")
        fig.update_layout(
            yaxis=dict(title=f"Revenue ({currency})"),
            yaxis2=dict(title="Bookings", overlaying="y", side="right"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        figures.append(fig)

    # 3. Payment status donut
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
        _apply_layout(fig, f"Revenue by Payment Status — {data['period']}", height=350)
        figures.append(fig)

    return figures


# ── Member insights ─────────────────────────────────────────────────────

def chart_members(data: dict) -> list[go.Figure]:
    figures = []

    # 1. Top bookers — horizontal bar
    top = data.get("top_bookers", [])
    if top:
        names = [b["name"] for b in reversed(top)]
        counts = [b["count"] for b in reversed(top)]

        fig = go.Figure(go.Bar(
            x=counts, y=names, orientation="h",
            marker_color=COLORS["primary"],
            text=counts, textposition="auto",
        ))
        _apply_layout(fig, f"Top Bookers — {data['period']}", height=max(300, len(top) * 40 + 100))
        fig.update_xaxes(title_text="Bookings")
        figures.append(fig)

    # 2. Level distribution — bar
    levels = data.get("padel_level_distribution", {})
    if levels:
        lvl_labels = list(levels.keys())
        lvl_counts = list(levels.values())

        fig = go.Figure(go.Bar(
            x=lvl_labels, y=lvl_counts,
            marker_color=COLORS["secondary"],
            text=lvl_counts, textposition="auto",
        ))
        _apply_layout(fig, "Padel Level Distribution")
        fig.update_xaxes(title_text="Level")
        fig.update_yaxes(title_text="Players")
        figures.append(fig)

    # 3. Active vs registered — gauge
    total = data.get("total_registered_players", 0)
    active = data.get("active_players_in_period", 0)
    if total > 0:
        pct = round(active / total * 100, 1)
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=pct,
            number={"suffix": "%"},
            title={"text": f"Active Players ({active}/{total})"},
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
        _apply_layout(fig, f"Active Players ({active}/{total})", height=280)
        figures.append(fig)

    return figures


# ── Operational alerts ──────────────────────────────────────────────────

def chart_operations(data: dict) -> list[go.Figure]:
    figures = []

    # 1. Peak hours bar chart
    peak = data.get("peak_hours", [])
    quiet = data.get("quiet_hours", [])
    all_hours_data = {}
    for h in peak + quiet:
        all_hours_data[h["hour"]] = h["bookings"]

    if all_hours_data:
        # Sort by hour
        sorted_h = sorted(all_hours_data.items(), key=lambda x: x[0])
        hours = [h[0] for h in sorted_h]
        counts = [h[1] for h in sorted_h]

        # Color: peak in green, quiet in red
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
        _apply_layout(fig, f"Bookings by Hour — {data['period']}")
        fig.update_xaxes(title_text="Hour (local time)")
        fig.update_yaxes(title_text="Bookings")
        figures.append(fig)

    # 2. Day of week distribution
    dow = data.get("day_of_week_distribution", {})
    if dow:
        day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        ordered = [(d, dow.get(d, 0)) for d in day_order if d in dow]
        days = [d[0][:3] for d in ordered]
        vals = [d[1] for d in ordered]

        fig = go.Figure(go.Bar(
            x=days, y=vals,
            marker_color=[COLORS["secondary"] if d[0] in ("Saturday", "Sunday") else COLORS["primary"] for d in ordered],
            text=vals, textposition="auto",
        ))
        _apply_layout(fig, f"Bookings by Day of Week — {data['period']}")
        fig.update_yaxes(title_text="Bookings")
        figures.append(fig)

    # 3. Booking type pie chart
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
        _apply_layout(fig, f"Booking Types — {data['period']}", height=350)
        figures.append(fig)

    # 4. Cancellation rate gauge
    cancel_rate = data.get("cancellation_rate_pct", 0)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=cancel_rate,
        number={"suffix": "%"},
        title={"text": "Cancellation Rate"},
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
    _apply_layout(fig, "Cancellation Rate", height=280)
    figures.append(fig)

    return figures


# ── Available slots ─────────────────────────────────────────────────────

def chart_available_slots(data: dict) -> list[go.Figure]:
    figures = []

    slots_by_court = data.get("slots_by_court", {})
    if not slots_by_court:
        return figures

    # 1. Available slots count per court
    courts = list(slots_by_court.keys())
    slot_counts = [len(s) for s in slots_by_court.values()]

    fig = go.Figure(go.Bar(
        x=courts, y=slot_counts,
        marker_color=COLORS["success"],
        text=slot_counts, textposition="auto",
    ))
    _apply_layout(fig, f"Available Slots per Court — {data['date']}")
    fig.update_yaxes(title_text="Open Slots")
    figures.append(fig)

    # 2. Availability timeline heatmap
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
            y=[f"Court {i+1}" for i in range(len(court_ids))],
            colorscale=[[0, "#ffebee"], [1, "#c8e6c9"]],
            showscale=False,
            zmin=0, zmax=1,
        ))
        _apply_layout(fig, f"Availability Heatmap — {data['date']}")
        fig.update_xaxes(title_text="Hour")
        fig.add_annotation(
            x=1.0, y=-0.15, xref="paper", yref="paper",
            text="Green = Available | Red = Booked",
            showarrow=False, font_size=11, font_color=COLORS["muted"],
        )
        figures.append(fig)

    return figures


# ── Dispatcher ──────────────────────────────────────────────────────────

def chart_booking_details(data: dict) -> list[go.Figure]:
    figures = []
    bookings = data.get("bookings", [])
    if not bookings:
        return figures

    # Court timeline with player names
    courts = {}
    for b in bookings:
        court = b.get("court", "Unknown")
        courts.setdefault(court, [])
        start_iso = b.get("start_iso", "")
        players = ", ".join(b.get("players", ["Unknown"]))
        if "T" in start_iso:
            h = int(start_iso.split("T")[1][:2])
            m = int(start_iso.split("T")[1][3:5])
            courts[court].append({"hour": h + m / 60, "players": players, "type": b.get("booking_type", "")})

    if courts:
        all_hours = list(range(7, 24))
        court_names = sorted(courts.keys())

        # Build heatmap matrix
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
            hovertemplate="<b>%{y}</b> at %{x}<br>Players: %{customdata}<extra></extra>",
            colorscale=[[0, "#f5f5f5"], [1, "#1B998B"]],
            showscale=False,
        ))
        filters = data.get("filters_applied", {})
        title_parts = [f"Booking Schedule — {data['date']}"]
        if filters.get("court_name"):
            title_parts.append(f"(Court: {filters['court_name']})")
        if filters.get("player_name"):
            title_parts.append(f"(Player: {filters['player_name']})")
        _apply_layout(fig, " ".join(title_parts))
        fig.update_xaxes(title_text="Hour (local time)")
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
    """Build charts for a given tool result. Returns empty list if no charts apply."""
    builder = CHART_BUILDERS.get(tool_name)
    if builder:
        try:
            return builder(tool_result)
        except Exception:
            return []
    return []
