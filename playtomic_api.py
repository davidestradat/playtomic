"""
Playtomic Third-Party API Client

Handles OAuth2 authentication, bookings, players, and availability queries
for the Playtomic club management platform.
"""

import time
from datetime import datetime, timedelta, timezone
from typing import Optional
from zoneinfo import ZoneInfo

import requests


class PlaytomicAPI:
    """Client for the Playtomic Third-Party API."""

    THIRD_PARTY_BASE = "https://thirdparty.playtomic.io/api/v1"
    PUBLIC_BASE = "https://api.playtomic.io/v1"

    def __init__(self, client_id: str, client_secret: str, tz_name: str = "America/Cancun"):
        self.client_id = client_id
        self.client_secret = client_secret
        self.club_tz = ZoneInfo(tz_name)
        self._token: Optional[str] = None
        self._token_expires_at: float = 0

    def _local_day_to_utc_range(self, local_date: datetime) -> tuple[datetime, datetime]:
        """
        Convert a local date to a UTC start/end range.
        e.g. Feb 19 local (UTC-5) -> Feb 19 05:00 UTC to Feb 20 04:59:59 UTC
        """
        local_start = local_date.replace(hour=0, minute=0, second=0, tzinfo=self.club_tz)
        local_end = local_date.replace(hour=23, minute=59, second=59, tzinfo=self.club_tz)
        utc_start = local_start.astimezone(timezone.utc).replace(tzinfo=None)
        utc_end = local_end.astimezone(timezone.utc).replace(tzinfo=None)
        return utc_start, utc_end

    # ── Authentication ──────────────────────────────────────────────────

    def _ensure_token(self):
        """Obtain or refresh the OAuth2 bearer token."""
        if self._token and time.time() < self._token_expires_at - 60:
            return
        resp = requests.post(
            f"{self.THIRD_PARTY_BASE}/oauth/token",
            json={"client_id": self.client_id, "secret": self.client_secret},
            headers={"Content-Type": "application/json"},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        self._token = data["token"]
        self._token_expires_at = time.time() + data.get("expires_in", 3600)

    def _headers(self) -> dict:
        self._ensure_token()
        return {"Authorization": f"Bearer {self._token}"}

    # ── Bookings ────────────────────────────────────────────────────────

    def get_bookings(
        self,
        tenant_id: str,
        start_date: datetime,
        end_date: datetime,
        booking_type: Optional[str] = None,
        sport_id: str = "PADEL",
        status: Optional[str] = None,
        page: int = 0,
        size: int = 200,
    ) -> list[dict]:
        """Fetch bookings for a tenant within a date range. Returns all pages."""
        all_bookings = []
        current_page = page

        while True:
            params = {
                "tenant_id": tenant_id,
                "start_booking_date": start_date.strftime("%Y-%m-%dT%H:%M:%S"),
                "end_booking_date": end_date.strftime("%Y-%m-%dT%H:%M:%S"),
                "sport_id": sport_id,
                "page": current_page,
                "size": size,
            }
            if booking_type:
                params["booking_type"] = booking_type
            if status:
                params["status"] = status

            resp = requests.get(
                f"{self.THIRD_PARTY_BASE}/bookings",
                params=params,
                headers=self._headers(),
                timeout=30,
            )
            resp.raise_for_status()
            bookings = resp.json()

            if not bookings:
                break

            all_bookings.extend(bookings)

            if len(bookings) < size:
                break
            current_page += 1

        return all_bookings

    def get_bookings_for_date(
        self, tenant_id: str, date: datetime, sport_id: str = "PADEL"
    ) -> list[dict]:
        """Get all bookings for a specific LOCAL date (converts to UTC for API)."""
        utc_start, utc_end = self._local_day_to_utc_range(date)
        return self.get_bookings(tenant_id, utc_start, utc_end, sport_id=sport_id)

    def get_bookings_for_range(
        self,
        tenant_id: str,
        start_date: datetime,
        end_date: datetime,
        sport_id: str = "PADEL",
    ) -> list[dict]:
        """Get all bookings for a LOCAL date range (converts to UTC for API)."""
        utc_start, _ = self._local_day_to_utc_range(start_date)
        _, utc_end = self._local_day_to_utc_range(end_date)
        return self.get_bookings(tenant_id, utc_start, utc_end, sport_id=sport_id)

    # ── Availability (Public API) ───────────────────────────────────────

    def get_availability(
        self,
        tenant_id: str,
        date: datetime,
        sport_id: str = "PADEL",
    ) -> list[dict]:
        """
        Get available (unbooked) court slots for a specific date.
        Uses the public Playtomic API (no auth required).
        Max 25-hour window per request.
        """
        start_min = date.replace(hour=0, minute=0, second=0)
        start_max = date.replace(hour=23, minute=59, second=59)

        params = {
            "tenant_id": tenant_id,
            "sport_id": sport_id,
            "local_start_min": start_min.strftime("%Y-%m-%dT%H:%M:%S"),
            "local_start_max": start_max.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        resp = requests.get(
            f"{self.PUBLIC_BASE}/availability",
            params=params,
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()

    # ── Players ─────────────────────────────────────────────────────────

    def get_players(
        self,
        venue_id: str,
        include: Optional[list[str]] = None,
        limit: int = 100,
    ) -> list[dict]:
        """Fetch all players for a venue using cursor pagination."""
        all_players = []
        cursor_id = None
        inc = include or ["BENEFITS", "SPORTS", "WALLETS"]

        while True:
            params = {"limit": limit, "include": ",".join(inc)}
            if cursor_id:
                params["cursor_id"] = cursor_id

            resp = requests.get(
                f"{self.THIRD_PARTY_BASE}/venues/{venue_id}/players",
                params=params,
                headers=self._headers(),
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()

            players = data.get("data", [])
            all_players.extend(players)

            if not data.get("has_more", False):
                break
            cursor_id = data.get("next_cursor_id")
            if not cursor_id:
                break

        return all_players

    def get_player(self, venue_id: str, player_id: str) -> dict:
        """Fetch a single player by ID."""
        resp = requests.get(
            f"{self.THIRD_PARTY_BASE}/venues/{venue_id}/players/{player_id}",
            headers=self._headers(),
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()

    # ── Venue Discovery (Public API) ────────────────────────────────────

    def discover_venues(
        self,
        sport_id: str = "PADEL",
        coordinate: Optional[dict] = None,
        radius: int = 50000,
        size: int = 20,
    ) -> list[dict]:
        """
        Search for venues via the public API.
        Useful for finding your tenant_id if you don't have it.
        """
        params = {
            "sport_id": sport_id,
            "playtomic_status": "ACTIVE",
            "size": size,
        }
        if coordinate:
            params["coordinate"] = f"{coordinate['lat']},{coordinate['lon']}"
            params["radius"] = radius

        resp = requests.get(
            f"{self.PUBLIC_BASE}/tenants",
            params=params,
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()

    def search_venues_by_name(self, name: str, sport_id: str = "PADEL") -> list[dict]:
        """Search venues by name fragment."""
        params = {
            "sport_id": sport_id,
            "playtomic_status": "ACTIVE",
            "tenant_name": name,
            "size": 50,
        }
        resp = requests.get(
            f"{self.PUBLIC_BASE}/tenants",
            params=params,
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()

    # ── Convenience: test connection ────────────────────────────────────

    def test_connection(self) -> bool:
        """Test that the credentials are valid by requesting a token."""
        try:
            self._ensure_token()
            return self._token is not None
        except Exception:
            return False
