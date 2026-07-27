"""
BusinessContextResolver — resolves city, state, and service BEFORE any LLM call.

Resolution priority (each step fills only what the previous steps left empty):

  1. Explicit values already on the ArticleRequest (CLI flags, API payload)
  2. SiteProfile config file  (profiles/{client_id}/{website_id}/site.json)
  3. WordPress REST API        (GET {url}/wp-json/ — public, no auth required)
  4. Website homepage HTML     (JSON-LD LocalBusiness schema, geo meta tags,
                                "City, ST" patterns in visible text)
  5. Article topic text        (regex extraction of "City, ST" from the topic string)
  6. URL domain heuristics     (substring city search in the normalised domain token)
  7. Warning + non-local article — caller is notified; publish gate enforces location

The planner and generator must never infer, invent, or copy business facts from
prompt examples.  This resolver guarantees that concrete, source-verified values
reach the planner via BUSINESS NICHE and TARGET LOCATION task-brief fields.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING

import httpx

from models.location import Location
from services.site_profile_service import SiteProfileService

if TYPE_CHECKING:
    from models.article import ArticleRequest

logger = logging.getLogger(__name__)

# ── HTTP settings ─────────────────────────────────────────────────────────────

_HTTP_TIMEOUT = 8.0      # seconds per request — short enough not to stall generation
_HTTP_HEADERS = {"User-Agent": "SEOAgent-ContextResolver/1.0"}

# ── State tables ──────────────────────────────────────────────────────────────

# Two-letter abbreviation → full name (used to validate regex captures)
_STATE_ABBREVS: frozenset[str] = frozenset({
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA",
    "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
    "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
    "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
    "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY", "DC",
})

# Full state name → abbreviation (lowercase key for case-insensitive match)
_STATE_NAMES: dict[str, str] = {
    "alabama": "AL", "alaska": "AK", "arizona": "AZ", "arkansas": "AR",
    "california": "CA", "colorado": "CO", "connecticut": "CT", "delaware": "DE",
    "florida": "FL", "georgia": "GA", "hawaii": "HI", "idaho": "ID",
    "illinois": "IL", "indiana": "IN", "iowa": "IA", "kansas": "KS",
    "kentucky": "KY", "louisiana": "LA", "maine": "ME", "maryland": "MD",
    "massachusetts": "MA", "michigan": "MI", "minnesota": "MN",
    "mississippi": "MS", "missouri": "MO", "montana": "MT", "nebraska": "NE",
    "nevada": "NV", "new hampshire": "NH", "new jersey": "NJ",
    "new mexico": "NM", "new york": "NY", "north carolina": "NC",
    "north dakota": "ND", "ohio": "OH", "oklahoma": "OK", "oregon": "OR",
    "pennsylvania": "PA", "rhode island": "RI", "south carolina": "SC",
    "south dakota": "SD", "tennessee": "TN", "texas": "TX", "utah": "UT",
    "vermont": "VT", "virginia": "VA", "washington": "WA",
    "west virginia": "WV", "wisconsin": "WI", "wyoming": "WY",
    "district of columbia": "DC", "washington dc": "DC",
}

# ── City → (display name, state abbreviation) ────────────────────────────────
# Keys: city name lowercased, spaces and hyphens removed.
# Searched as substrings inside normalised domain tokens (domain heuristics only).
# For free-text extraction, _CITY_STATE_RE is preferred.

_CITIES: dict[str, tuple[str, str]] = {
    # A
    "akron": ("Akron", "OH"),
    "albuquerque": ("Albuquerque", "NM"),
    "alexandria": ("Alexandria", "VA"),
    "anaheim": ("Anaheim", "CA"),
    "anchorage": ("Anchorage", "AK"),
    "annarbor": ("Ann Arbor", "MI"),
    "arlington": ("Arlington", "TX"),
    "atlanta": ("Atlanta", "GA"),
    "augusta": ("Augusta", "GA"),
    "aurora": ("Aurora", "CO"),
    "austin": ("Austin", "TX"),
    # B
    "bakersfield": ("Bakersfield", "CA"),
    "batonrouge": ("Baton Rouge", "LA"),
    "bellevue": ("Bellevue", "WA"),
    "birmingham": ("Birmingham", "AL"),
    "bismarck": ("Bismarck", "ND"),
    "boise": ("Boise", "ID"),
    "boston": ("Boston", "MA"),
    "bowlinggreen": ("Bowling Green", "KY"),
    "bridgeport": ("Bridgeport", "CT"),
    "buffalo": ("Buffalo", "NY"),
    # C
    "capecoral": ("Cape Coral", "FL"),
    "cedar rapids": ("Cedar Rapids", "IA"),
    "cedarrapids": ("Cedar Rapids", "IA"),
    "chandler": ("Chandler", "AZ"),
    "charleston": ("Charleston", "SC"),
    "charlotte": ("Charlotte", "NC"),
    "chattanooga": ("Chattanooga", "TN"),
    "chesapeake": ("Chesapeake", "VA"),
    "chicago": ("Chicago", "IL"),
    "cincinnati": ("Cincinnati", "OH"),
    "clarksville": ("Clarksville", "TN"),
    "cleveland": ("Cleveland", "OH"),
    "clovis": ("Clovis", "CA"),
    "coloradosprings": ("Colorado Springs", "CO"),
    "columbia": ("Columbia", "SC"),
    "columbus": ("Columbus", "OH"),
    "corona": ("Corona", "CA"),
    "corpuschristi": ("Corpus Christi", "TX"),
    # D
    "dallas": ("Dallas", "TX"),
    "dayton": ("Dayton", "OH"),
    "desmoines": ("Des Moines", "IA"),
    "durham": ("Durham", "NC"),
    # E
    "elpaso": ("El Paso", "TX"),
    "elkgrove": ("Elk Grove", "CA"),
    "escondido": ("Escondido", "CA"),
    "eugene": ("Eugene", "OR"),
    "evansville": ("Evansville", "IN"),
    # F
    "fayetteville": ("Fayetteville", "NC"),
    "flint": ("Flint", "MI"),
    "fontana": ("Fontana", "CA"),
    "fortlauderdale": ("Fort Lauderdale", "FL"),
    "fortsmith": ("Fort Smith", "AR"),
    "fortwayne": ("Fort Wayne", "IN"),
    "fortworth": ("Fort Worth", "TX"),
    "fremont": ("Fremont", "CA"),
    "fresno": ("Fresno", "CA"),
    # G
    "gainesville": ("Gainesville", "FL"),
    "gardengrove": ("Garden Grove", "CA"),
    "garland": ("Garland", "TX"),
    "gilbert": ("Gilbert", "AZ"),
    "glendale": ("Glendale", "AZ"),
    "grandrapids": ("Grand Rapids", "MI"),
    "greensboro": ("Greensboro", "NC"),
    "greenville": ("Greenville", "SC"),
    # H
    "hampton": ("Hampton", "VA"),
    "hartford": ("Hartford", "CT"),
    "hayward": ("Hayward", "CA"),
    "henderson": ("Henderson", "NV"),
    "hialeah": ("Hialeah", "FL"),
    "hollywood": ("Hollywood", "FL"),
    "honolulu": ("Honolulu", "HI"),
    "houston": ("Houston", "TX"),
    "huntsville": ("Huntsville", "AL"),
    # I
    "indianapolis": ("Indianapolis", "IN"),
    "irvine": ("Irvine", "CA"),
    # J
    "jackson": ("Jackson", "MS"),
    "jacksonville": ("Jacksonville", "FL"),
    "jerseycity": ("Jersey City", "NJ"),
    # K
    "kalamazoo": ("Kalamazoo", "MI"),
    "kansascity": ("Kansas City", "MO"),
    "killeen": ("Killeen", "TX"),
    "knoxville": ("Knoxville", "TN"),
    # L
    "lakewood": ("Lakewood", "CO"),
    "lancaster": ("Lancaster", "CA"),
    "lansing": ("Lansing", "MI"),
    "laredo": ("Laredo", "TX"),
    "lasvegas": ("Las Vegas", "NV"),
    "lexington": ("Lexington", "KY"),
    "lincoln": ("Lincoln", "NE"),
    "littlerock": ("Little Rock", "AR"),
    "longbeach": ("Long Beach", "CA"),
    "losangeles": ("Los Angeles", "CA"),
    "louisville": ("Louisville", "KY"),
    "lubbock": ("Lubbock", "TX"),
    # M
    "macon": ("Macon", "GA"),
    "madison": ("Madison", "WI"),
    "mcallen": ("McAllen", "TX"),
    "memphis": ("Memphis", "TN"),
    "mesa": ("Mesa", "AZ"),
    "mesquite": ("Mesquite", "TX"),
    "miami": ("Miami", "FL"),
    "milwaukee": ("Milwaukee", "WI"),
    "minneapolis": ("Minneapolis", "MN"),
    "modesto": ("Modesto", "CA"),
    "morenovalley": ("Moreno Valley", "CA"),
    # N
    "naperville": ("Naperville", "IL"),
    "nashville": ("Nashville", "TN"),
    "neworleans": ("New Orleans", "LA"),
    "newyork": ("New York", "NY"),
    "norfolk": ("Norfolk", "VA"),
    "northlasvega": ("North Las Vegas", "NV"),
    # O
    "oakland": ("Oakland", "CA"),
    "oceanside": ("Oceanside", "CA"),
    "oklahomacity": ("Oklahoma City", "OK"),
    "omaha": ("Omaha", "NE"),
    "orange": ("Orange", "CA"),
    "orlando": ("Orlando", "FL"),
    "owensboro": ("Owensboro", "KY"),
    "oxnard": ("Oxnard", "CA"),
    # P
    "palmdale": ("Palmdale", "CA"),
    "pasadena": ("Pasadena", "TX"),
    "paterson": ("Paterson", "NJ"),
    "peoria": ("Peoria", "IL"),
    "philadelphia": ("Philadelphia", "PA"),
    "phoenix": ("Phoenix", "AZ"),
    "pittsburgh": ("Pittsburgh", "PA"),
    "plano": ("Plano", "TX"),
    "pomona": ("Pomona", "CA"),
    "portland": ("Portland", "OR"),
    "providence": ("Providence", "RI"),
    # R
    "raleigh": ("Raleigh", "NC"),
    "rapidcity": ("Rapid City", "SD"),
    "reno": ("Reno", "NV"),
    "richmond": ("Richmond", "VA"),
    "riverside": ("Riverside", "CA"),
    "rockford": ("Rockford", "IL"),
    "rochester": ("Rochester", "NY"),
    # S
    "sacramento": ("Sacramento", "CA"),
    "salinas": ("Salinas", "CA"),
    "sanantonio": ("San Antonio", "TX"),
    "sandiego": ("San Diego", "CA"),
    "sanfrancisco": ("San Francisco", "CA"),
    "sanjose": ("San Jose", "CA"),
    "santaana": ("Santa Ana", "CA"),
    "santaclarita": ("Santa Clarita", "CA"),
    "savannah": ("Savannah", "GA"),
    "scottsdale": ("Scottsdale", "AZ"),
    "seattle": ("Seattle", "WA"),
    "shreveport": ("Shreveport", "LA"),
    "siouxfalls": ("Sioux Falls", "SD"),
    "spokane": ("Spokane", "WA"),
    "springfield": ("Springfield", "MO"),
    "sterlingheights": ("Sterling Heights", "MI"),
    "stockton": ("Stockton", "CA"),
    "stlouis": ("St. Louis", "MO"),
    "stpaul": ("St. Paul", "MN"),
    "sunnyvale": ("Sunnyvale", "CA"),
    "sunrise": ("Sunrise", "FL"),
    "syracuse": ("Syracuse", "NY"),
    # T
    "tacoma": ("Tacoma", "WA"),
    "tallahassee": ("Tallahassee", "FL"),
    "tampa": ("Tampa", "FL"),
    "tempe": ("Tempe", "AZ"),
    "thousandoaks": ("Thousand Oaks", "CA"),
    "toledo": ("Toledo", "OH"),
    "topeka": ("Topeka", "KS"),
    "torrance": ("Torrance", "CA"),
    "tucson": ("Tucson", "AZ"),
    # V
    "vallejo": ("Vallejo", "CA"),
    "virginia": ("Virginia Beach", "VA"),
    "visalia": ("Visalia", "CA"),
    # W
    "washington": ("Washington", "DC"),
    "waterbury": ("Waterbury", "CT"),
    "wichita": ("Wichita", "KS"),
    "winston": ("Winston-Salem", "NC"),
    "worcester": ("Worcester", "MA"),
    # Y
    "yonkers": ("Yonkers", "NY"),
}

# Sorted by key length descending — longer keys shadow shorter ones in substring search.
_CITIES_BY_LENGTH: list[tuple[str, tuple[str, str]]] = sorted(
    _CITIES.items(), key=lambda kv: len(kv[0]), reverse=True,
)

# ── Text-based city/state extraction ─────────────────────────────────────────

# "Buffalo, NY" or "Fort Worth, TX" — 1–4 title-case words, comma, 2-letter state
_CITY_STATE_ABBREV_RE = re.compile(
    r'\b([A-Z][a-z]+(?: [A-Z][a-z]+){0,3}),\s*([A-Z]{2})\b'
)

# "Buffalo, New York" — city followed by a full state name
_CITY_STATE_FULL_RE = re.compile(
    r'\b([A-Z][a-z]+(?: [A-Z][a-z]+){0,3}),\s*'
    r'((?:New |North |South |West |Rhode )?[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b'
)

# Finds state abbreviations in text so we can look backward for the city.
# Only matches exactly 2 uppercase letters at a word boundary — won't match
# lowercase "in", "or", "oh" but will match "IN", "OR", "OH".
_STATE_ABBREV_IN_TEXT_RE = re.compile(r'\b([A-Z]{2})\b')

# Title-case word(s) at end of a string — used for backward city lookup.
_TRAILING_TITLE_WORDS_RE = re.compile(r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s*$')

# ── Service keyword extraction ────────────────────────────────────────────────

_SERVICE_KEYWORDS: list[tuple[str, str]] = [
    ("overhead door", "Overhead Door"),
    ("garage door", "Garage Door Repair"),
    ("garagedoor", "Garage Door Repair"),
    ("plumbing", "Plumbing"),
    ("hvac", "HVAC"),
    ("heating and cooling", "HVAC"),
    ("heating & cooling", "HVAC"),
    ("air conditioning", "HVAC"),
    ("roofing", "Roofing"),
    ("electrician", "Electrical"),
    ("electrical", "Electrical"),
    ("locksmith", "Locksmith"),
    ("pest control", "Pest Control"),
    ("landscaping", "Landscaping"),
    ("window installation", "Window Installation"),
    ("flooring", "Flooring"),
    ("painting", "Painting"),
    ("handyman", "Handyman"),
]

# ── URL normalisation ─────────────────────────────────────────────────────────

_PROTOCOL_RE = re.compile(r'^https?://', re.IGNORECASE)
_TLD_RE = re.compile(r'\.(com|net|org|us|biz|info|co|io)$', re.IGNORECASE)


def _normalise_domain(url: str) -> str:
    """Return the bare domain token: no protocol, www, path, port, or TLD."""
    raw = _PROTOCOL_RE.sub("", url).lower()
    raw = raw.split("/")[0].split(":")[0]
    raw = raw.lstrip("www.")
    raw = _TLD_RE.sub("", raw)
    return raw.replace("-", "")


# ── Extraction helpers ────────────────────────────────────────────────────────

def _city_state_from_text(text: str) -> Location | None:
    """
    Extract (city, state) from free-form text.

    Tries in order:
      a) "City, ST"        — explicit abbreviation after comma (most reliable)
      b) "City, Full State"— full state name after comma
      c) "City ST"         — no comma; city validated against cities table
                             (catches topic strings like "Bowling Green KY")
    """
    # (a) explicit abbreviation with comma
    for match in _CITY_STATE_ABBREV_RE.finditer(text):
        city = match.group(1).strip()
        state = match.group(2).upper()
        if state in _STATE_ABBREVS:
            return Location(city=city, state=state)

    # (b) full state name with comma
    for match in _CITY_STATE_FULL_RE.finditer(text):
        city = match.group(1).strip()
        state_name = match.group(2).strip().lower()
        abbrev = _STATE_NAMES.get(state_name)
        if abbrev:
            return Location(city=city, state=abbrev)

    # (c) no comma — find state abbreviations, then look *backward* for the city.
    # Backward lookup avoids the greedy-match bug where "Repair Buffalo NY"
    # consumes "Buffalo" before it can be tested as a city candidate.
    for state_match in _STATE_ABBREV_IN_TEXT_RE.finditer(text):
        state = state_match.group(1)
        if state not in _STATE_ABBREVS:
            continue
        prefix = text[: state_match.start()].rstrip()
        title_match = _TRAILING_TITLE_WORDS_RE.search(prefix)
        if not title_match:
            continue
        words = title_match.group(1).split()
        # Try 2-word city first (e.g. "Bowling Green"), then 1-word (e.g. "Buffalo")
        for n in (2, 1):
            if len(words) >= n:
                candidate = " ".join(words[-n:])
                city_key = re.sub(r"[\s\-]", "", candidate.lower())
                if city_key in _CITIES:
                    return Location(city=candidate, state=state)

    return None


def _service_from_text(text: str) -> str | None:
    """Extract a service category from free-form text using keyword matching."""
    lower = text.lower()
    for keyword, service in _SERVICE_KEYWORDS:
        if keyword in lower:
            return service
    return None


def _city_from_domain(url: str) -> Location | None:
    """
    Extract city from a website URL using substring matching against the cities table.

    Searches the normalised domain token (no spaces, no hyphens, no TLD) for any
    known city key. Longer keys are checked first so "sandiego" wins over "san".
    """
    token = _normalise_domain(url)
    for key, (city_name, state) in _CITIES_BY_LENGTH:
        if key in token:
            logger.info(
                "Domain heuristic: %r contains %r → %r, %r", token, key, city_name, state
            )
            return Location(city=city_name, state=state)
    logger.debug("Domain heuristic: no city found in %r", token)
    return None


def _fetch_wp_root(url: str) -> dict[str, str]:
    """
    Fetch the public WordPress REST API root endpoint.

    Returns a dict with 'name' and 'description' (both str, may be empty).
    Returns {} on any network or parse error — callers treat it as no data.
    """
    try:
        endpoint = url.rstrip("/") + "/wp-json/"
        resp = httpx.get(endpoint, timeout=_HTTP_TIMEOUT, headers=_HTTP_HEADERS,
                          follow_redirects=True)
        if resp.status_code == 200:
            data = resp.json()
            return {
                "name": data.get("name") or "",
                "description": data.get("description") or "",
                "url": data.get("url") or "",
            }
        logger.debug("WP root returned %d for %r", resp.status_code, endpoint)
    except Exception as exc:
        logger.debug("WP root fetch failed for %r: %s", url, exc)
    return {}


def _fetch_html(url: str, path: str = "/") -> str:
    """
    Fetch a page from the website and return its HTML body.

    Returns "" on any error — callers treat it as no data.
    """
    try:
        target = url.rstrip("/") + path
        resp = httpx.get(target, timeout=_HTTP_TIMEOUT, headers=_HTTP_HEADERS,
                          follow_redirects=True)
        if resp.status_code == 200:
            return resp.text
        logger.debug("HTML fetch returned %d for %r", resp.status_code, target)
    except Exception as exc:
        logger.debug("HTML fetch failed for %r%r: %s", url, path, exc)
    return ""


def _extract_from_jsonld(html: str) -> tuple[Location | None, str | None]:
    """
    Parse JSON-LD blocks in HTML for LocalBusiness / Organization schema.

    Returns (location, service) — either may be None if not found.
    """
    scripts = re.findall(
        r'<script[^>]+type=["\']application/ld\+json["\'][^>]*>(.*?)</script>',
        html, re.DOTALL | re.IGNORECASE,
    )

    location: Location | None = None
    service: str | None = None

    for script in scripts:
        try:
            data = json.loads(script.strip())
        except (json.JSONDecodeError, ValueError):
            continue

        items: list = []
        if isinstance(data, list):
            items = data
        elif isinstance(data, dict):
            items = data.get("@graph", [data])

        for item in items:
            if not isinstance(item, dict):
                continue
            schema_type = str(item.get("@type", ""))
            if not any(t in schema_type for t in (
                "LocalBusiness", "Organization", "HomeAndConstructionBusiness",
                "Store", "Service", "Plumber", "HVACBusiness", "Electrician",
                "RoofingContractor", "GeneralContractor",
            )):
                continue

            # City + state from address block
            if location is None:
                address = item.get("address", {})
                if isinstance(address, dict):
                    locality = address.get("addressLocality") or address.get("addressCity") or ""
                    region = address.get("addressRegion") or address.get("addressState") or ""
                    if locality and region:
                        state_abbrev = region.upper() if len(region) == 2 else _STATE_NAMES.get(region.lower())
                        if state_abbrev and state_abbrev in _STATE_ABBREVS:
                            location = Location(city=locality.strip(), state=state_abbrev)

            # Service from schema type or name
            if service is None:
                name = item.get("name") or ""
                desc = item.get("description") or ""
                service = _service_from_text(name) or _service_from_text(desc) or _service_from_text(schema_type)

    return location, service


def _extract_from_meta_tags(html: str) -> Location | None:
    """Extract city/state from HTML geo meta tags."""
    # <meta name="geo.placename" content="Buffalo, NY">
    patterns = [
        re.compile(r'<meta[^>]+name=["\']geo\.placename["\'][^>]+content=["\']([^"\']+)["\']', re.IGNORECASE),
        re.compile(r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+name=["\']geo\.placename["\']', re.IGNORECASE),
    ]
    for pat in patterns:
        m = pat.search(html)
        if m:
            loc = _city_state_from_text(m.group(1))
            if loc:
                return loc
    return None


def _extract_from_html_text(html: str) -> Location | None:
    """
    Look for "City, ST" pattern in key HTML sections.

    Checks (in order): footer, address tags, header, then full page text.
    Stops at the first confident match.
    """
    sections: list[str] = []

    # 1. <address> elements — most semantically reliable
    for m in re.finditer(r'<address[^>]*>(.*?)</address>', html, re.DOTALL | re.IGNORECASE):
        sections.append(m.group(1))

    # 2. <footer> — business info is almost always here
    for m in re.finditer(r'<footer[^>]*>(.*?)</footer>', html, re.DOTALL | re.IGNORECASE):
        sections.append(m.group(1))

    # 3. <header> — sometimes shows city in tagline
    for m in re.finditer(r'<header[^>]*>(.*?)</header>', html, re.DOTALL | re.IGNORECASE):
        sections.append(m.group(1))

    # 4. Full page as fallback
    sections.append(html)

    for section in sections:
        # Strip tags for clean text matching
        text = re.sub(r'<[^>]+>', ' ', section)
        text = re.sub(r'\s+', ' ', text)
        loc = _city_state_from_text(text)
        if loc:
            return loc

    return None


# ── Main resolver ─────────────────────────────────────────────────────────────

class BusinessContextResolver:
    """
    Resolves city, state, and service from configured and live sources before
    any LLM call.

    The planner must never infer, invent, or copy business facts from prompt
    examples.  This resolver guarantees concrete, source-verified values reach
    the planner as BUSINESS NICHE and TARGET LOCATION task-brief fields.

    Resolution priority
    -------------------
    1. Explicit values on the ArticleRequest (CLI flags, API payload)
    2. SiteProfile config file
    3. WordPress REST API (public /wp-json/ endpoint)
    4. Website homepage HTML (JSON-LD schema, geo meta tags, address/footer text)
    5. Article topic text (regex extraction — deterministic, not LLM)
    6. URL domain heuristics (substring city search)
    7. Warning — returns request without location; publish gate blocks if still unresolved
    """

    def __init__(self, profiles_dir: Path) -> None:
        self._profiles = SiteProfileService(profiles_dir)

    def resolve(
        self,
        client_id: str,
        website_id: str,
        request: "ArticleRequest",
    ) -> "ArticleRequest":
        """
        Return a copy of *request* with location and service filled in.

        City + state are attempted — logs a warning if they cannot
        be determined from any source.  Service is resolved where possible but is
        not a hard requirement (the planner infers it from the article topic when
        absent).
        """
        city: str | None = request.location.city if request.location else None
        state: str | None = request.location.state if request.location else None
        service: str | None = request.service
        source = "request"

        # ── 2. SiteProfile ────────────────────────────────────────────────────
        profile = self._profiles.load(client_id, website_id)
        if profile:
            if not city:
                city = profile.city or None
            if not state:
                state = profile.state or None
            if not service:
                service = profile.primary_service or None
            source = "site_profile"
            logger.info(
                "Context from site_profile: city=%r  state=%r  service=%r",
                city, state, service,
            )

        url = request.website_url

        if url and (not city or not state or not service):
            # ── 3. WordPress REST API ─────────────────────────────────────────
            logger.debug("Fetching WP metadata from %r", url)
            wp = _fetch_wp_root(url)
            wp_text = f"{wp.get('name', '')} {wp.get('description', '')}"

            if wp_text.strip():
                if not city or not state:
                    loc = _city_state_from_text(wp_text)
                    if loc:
                        if not city:
                            city = loc.city
                        if not state:
                            state = loc.state
                        source = "wp_api"
                        logger.info(
                            "Context from wp_api (%r): city=%r  state=%r",
                            wp_text.strip(), city, state,
                        )
                if not service:
                    service = _service_from_text(wp_text)

        if url and (not city or not state or not service):
            # ── 4. Homepage HTML ──────────────────────────────────────────────
            logger.debug("Fetching homepage HTML from %r", url)
            html = _fetch_html(url)

            if html:
                # JSON-LD LocalBusiness schema — highest confidence in HTML
                if not city or not state or not service:
                    jld_loc, jld_svc = _extract_from_jsonld(html)
                    if jld_loc and (not city or not state):
                        if not city:
                            city = jld_loc.city
                        if not state:
                            state = jld_loc.state
                        source = "jsonld_schema"
                        logger.info(
                            "Context from JSON-LD: city=%r  state=%r", city, state
                        )
                    if jld_svc and not service:
                        service = jld_svc

                # Geo meta tags
                if not city or not state:
                    meta_loc = _extract_from_meta_tags(html)
                    if meta_loc:
                        if not city:
                            city = meta_loc.city
                        if not state:
                            state = meta_loc.state
                        source = "geo_meta"
                        logger.info(
                            "Context from geo meta: city=%r  state=%r", city, state
                        )

                # Address / footer / visible text
                if not city or not state:
                    text_loc = _extract_from_html_text(html)
                    if text_loc:
                        if not city:
                            city = text_loc.city
                        if not state:
                            state = text_loc.state
                        source = "html_text"
                        logger.info(
                            "Context from HTML text: city=%r  state=%r", city, state
                        )

        # ── 5. Topic text ─────────────────────────────────────────────────────
        if (not city or not state) and request.topic:
            topic_loc = _city_state_from_text(request.topic)
            if topic_loc:
                if not city:
                    city = topic_loc.city
                if not state:
                    state = topic_loc.state
                source = "topic_text"
                logger.info(
                    "Context from topic text: city=%r  state=%r", city, state
                )

        # ── 6. Domain heuristics ──────────────────────────────────────────────
        if (not city or not state) and url:
            domain_loc = _city_from_domain(url)
            if domain_loc:
                if not city:
                    city = domain_loc.city
                if not state:
                    state = domain_loc.state
                source = "domain_heuristic"
                logger.info(
                    "Context from domain heuristic: city=%r  state=%r", city, state
                )

        # ── 7. Location unresolvable — warn and continue without location ────────
        # Generation is still possible as a non-local article.
        # The publish pipeline enforces location as a hard requirement.
        if not city or not state:
            hint = (
                f"seo profile create --client {client_id} --website {website_id}"
                if client_id and website_id
                else "provide --city and --state on the command line"
            )
            logger.warning(
                "Location could not be resolved for %r/%r — "
                "article will generate without geographic targeting. "
                "To enable local SEO: %s",
                client_id, website_id, hint,
            )
            if service and not request.service:
                return request.model_copy(update={"service": service})
            return request

        if not service:
            logger.info(
                "Service not resolved for %r/%r — planner will infer from topic.",
                client_id, website_id,
            )

        logger.info(
            "Business context resolved (%s): city=%r  state=%r  service=%r",
            source, city, state, service,
        )
        return request.model_copy(update={
            "location": Location(city=city, state=state),
            "service": service,
        })
