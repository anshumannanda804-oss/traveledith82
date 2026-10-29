import os
import requests
import xml.etree.ElementTree as ET
import time
from email.utils import parsedate_to_datetime

# SACHET Configuration
SACHET_RSS_URL = os.getenv(
    "SACHET_RSS_URL",
    "https://sachet.ndma.gov.in/cap_public_website/rss/rss_india.xml",
).strip()
SACHET_CACHE_TTL_SECONDS = 300
SACHET_MAX_ALERTS_RETURNED = 8
SACHET_NEARBY_RADIUS_KM = 75
CAP_NS = {"cap": "urn:oasis:names:tc:emergency:cap:1.2"}

_sachet_feed_cache = {"expires_at": 0, "items": [], "error": None}
_sachet_detail_cache = {}
_sachet_polygon_cache = {}
IMD_API_KEY = os.getenv("IMD_API_KEY", "").strip()

from travel_utils import (
    calculate_distance, 
    simplify_match_text, 
    text_matches_terms, 
    is_point_in_india_bounds, 
    parse_rss_date
)

def reverse_geocode_for_alerts(lat, lon):
    try:
        response = requests.get(
            "https://nominatim.openstreetmap.org/reverse",
            params={"lat": lat, "lon": lon, "format": "json", "addressdetails": 1},
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36"},
            timeout=5,
        )
        if response.status_code != 200:
            return {}
        return response.json().get("address", {}) or {}
    except (requests.exceptions.RequestException, ValueError) as error:
        print(f"Alert reverse-geocode error: {error}")
        return {}


def build_alert_location_terms(destination_name, address):
    stop_words = {
        "india", "district", "division", "tehsil", "taluk", "state", "city",
        "municipal", "corporation", "urban", "rural", "subdistrict",
    }
    candidates = [
        destination_name,
        address.get("city"),
        address.get("town"),
        address.get("village"),
        address.get("suburb"),
        address.get("county"),
        address.get("state_district"),
        address.get("district"),
        address.get("state"),
    ]
    terms = set()
    for candidate in candidates:
        simple_candidate = simplify_match_text(candidate)
        if not simple_candidate or simple_candidate in stop_words:
            continue
        terms.add(simple_candidate)
        for token in simple_candidate.split():
            if len(token) >= 4 and token not in stop_words:
                terms.add(token)
    return terms


def fetch_sachet_feed_items():
    now = time.time()
    if _sachet_feed_cache["expires_at"] > now:
        return _sachet_feed_cache["items"], _sachet_feed_cache["error"]

    if not SACHET_RSS_URL:
        return [], "Sachet RSS URL is not configured."

    try:
        response = requests.get(
            SACHET_RSS_URL,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36"},
            timeout=15,
        )
        response.raise_for_status()
        root = ET.fromstring(response.content)
        items = []
        for item in root.findall("./channel/item"):
            items.append({
                "title": item.findtext("title", default="").strip(),
                "description": item.findtext("description", default="").strip(),
                "category": item.findtext("category", default="").strip(),
                "link": item.findtext("link", default="").strip(),
                "author": item.findtext("author", default="").strip(),
                "guid": item.findtext("guid", default="").strip(),
                "published": parse_rss_date(item.findtext("pubDate", default="")),
            })
        _sachet_feed_cache.update({
            "expires_at": now + SACHET_CACHE_TTL_SECONDS,
            "items": items,
            "error": None,
        })
        return items, None
    except (requests.exceptions.RequestException, ET.ParseError) as error:
        message = f"Sachet alert feed could not be reached: {error}"
        print(message)
        _sachet_feed_cache.update({
            "expires_at": now + 60,
            "items": [],
            "error": message,
        })
        return [], message


def cap_find_text(parent, path):
    if parent is None:
        return ""
    return (parent.findtext(path, default="", namespaces=CAP_NS) or "").strip()


def parse_sachet_cap_detail(link):
    if not link:
        return {}
    if link in _sachet_detail_cache:
        return _sachet_detail_cache[link]

    try:
        response = requests.get(
            link,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36"},
            timeout=10,
        )
        response.raise_for_status()
        root = ET.fromstring(response.content)
    except (requests.exceptions.RequestException, ET.ParseError) as error:
        print(f"Sachet CAP detail error: {error}")
        return {}

    infos = root.findall("cap:info", CAP_NS)
    info = next(
        (entry for entry in infos if cap_find_text(entry, "cap:language").lower().startswith("en")),
        infos[0] if infos else None,
    )
    area = info.find("cap:area", CAP_NS) if info is not None else None

    polygon_url = ""
    if info is not None:
        for parameter in info.findall("cap:parameter", CAP_NS):
            name = cap_find_text(parameter, "cap:valueName").casefold()
            value = cap_find_text(parameter, "cap:value")
            if name == "polygon url" and value:
                polygon_url = value
                break

    detail = {
        "identifier": cap_find_text(root, "cap:identifier"),
        "sender": cap_find_text(root, "cap:sender"),
        "sent": cap_find_text(root, "cap:sent"),
        "category": cap_find_text(info, "cap:category"),
        "event": cap_find_text(info, "cap:event"),
        "urgency": cap_find_text(info, "cap:urgency"),
        "severity": cap_find_text(info, "cap:severity") or "Unknown",
        "certainty": cap_find_text(info, "cap:certainty"),
        "effective": cap_find_text(info, "cap:effective"),
        "onset": cap_find_text(info, "cap:onset"),
        "expires": cap_find_text(info, "cap:expires"),
        "headline": cap_find_text(info, "cap:headline"),
        "description": cap_find_text(info, "cap:description"),
        "instruction": cap_find_text(info, "cap:instruction"),
        "area": cap_find_text(area, "cap:areaDesc"),
        "polygon_url": polygon_url,
        "source": "SACHET/NDMA CAP",
    }
    _sachet_detail_cache[link] = detail
    return detail


def fetch_sachet_polygon_points(polygon_url):
    if not polygon_url:
        return []
    if polygon_url in _sachet_polygon_cache:
        return _sachet_polygon_cache[polygon_url]

    points = []
    try:
        response = requests.get(
            polygon_url,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36"},
            timeout=10,
        )
        response.raise_for_status()
        root = ET.fromstring(response.content)
        for polygon in root.findall(".//polygon"):
            for raw_pair in (polygon.text or "").split():
                try:
                    lat_value, lon_value = raw_pair.split(",", 1)
                    points.append((float(lat_value), float(lon_value)))
                except (ValueError, TypeError):
                    continue
    except (requests.exceptions.RequestException, ET.ParseError) as error:
        print(f"Sachet polygon error: {error}")

    _sachet_polygon_cache[polygon_url] = points
    return points


def point_in_polygon(lat, lon, polygon):
    if len(polygon) < 3:
        return False

    x = lon
    y = lat
    inside = False
    previous_lat, previous_lon = polygon[-1]

    for current_lat, current_lon in polygon:
        y1 = previous_lat
        x1 = previous_lon
        y2 = current_lat
        x2 = current_lon
        intersects = ((y1 > y) != (y2 > y)) and (
            x < (x2 - x1) * (y - y1) / ((y2 - y1) or 1e-12) + x1
        )
        if intersects:
            inside = not inside
        previous_lat, previous_lon = current_lat, current_lon

    return inside


def min_distance_to_polygon_vertices(lat, lon, polygon):
    if not polygon:
        return None
    return min(calculate_distance(lat, lon, point_lat, point_lon) for point_lat, point_lon in polygon)


def match_sachet_alert_to_location(alert, lat, lon, location_terms):
    text_blob = " ".join([
        alert.get("area", ""),
        alert.get("headline", ""),
        alert.get("description", ""),
        alert.get("sender", ""),
    ])
    matched_term = text_matches_terms(text_blob, location_terms)
    if matched_term:
        return "location text", None

    polygon = fetch_sachet_polygon_points(alert.get("polygon_url"))
    if polygon:
        lat_values = [point[0] for point in polygon]
        lon_values = [point[1] for point in polygon]
        if (
            min(lat_values) - 1 <= lat <= max(lat_values) + 1
            and min(lon_values) - 1 <= lon <= max(lon_values) + 1
        ):
            if point_in_polygon(lat, lon, polygon):
                return "alert polygon", 0
            distance = min_distance_to_polygon_vertices(lat, lon, polygon)
            if distance is not None and distance <= SACHET_NEARBY_RADIUS_KM:
                return "nearby alert polygon", int(distance)

    return None, None


def fetch_india_sachet_alerts(lat, lon, destination_name=""):
    if not is_point_in_india_bounds(lat, lon):
        return [], {
            "name": "SACHET/NDMA CAP RSS",
            "status": "outside_india_bounds",
            "url": SACHET_RSS_URL,
            "imd_api_key_configured": bool(IMD_API_KEY),
        }

    address = reverse_geocode_for_alerts(lat, lon)
    country_code = (address.get("country_code") or "").lower()
    if country_code and country_code != "in":
        return [], {
            "name": "SACHET/NDMA CAP RSS",
            "status": "outside_india",
            "url": SACHET_RSS_URL,
            "imd_api_key_configured": bool(IMD_API_KEY),
        }

    location_terms = build_alert_location_terms(destination_name, address)
    feed_items, feed_error = fetch_sachet_feed_items()
    if feed_error:
        return [], {
            "name": "SACHET/NDMA CAP RSS",
            "status": "unavailable",
            "message": feed_error,
            "url": SACHET_RSS_URL,
            "imd_api_key_configured": bool(IMD_API_KEY),
        }

    matches = []
    checked_non_text_candidates = 0
    for index, item in enumerate(feed_items):
        item_text = " ".join([
            item.get("title", ""),
            item.get("description", ""),
            item.get("author", ""),
            item.get("category", ""),
        ])
        item_term_match = text_matches_terms(item_text, location_terms)
        should_probe_detail = bool(item_term_match)

        if not should_probe_detail and checked_non_text_candidates < 12 and index < 25:
            should_probe_detail = True
            checked_non_text_candidates += 1

        if not should_probe_detail:
            continue

        detail = parse_sachet_cap_detail(item.get("link", ""))
        if not detail:
            continue

        match_type, distance = match_sachet_alert_to_location(
            detail,
            float(lat),
            float(lon),
            location_terms,
        )
        if not match_type:
            continue

        headline = detail.get("headline") or item.get("title") or "India alert"
        matches.append({
            "name": headline,
            "event": detail.get("event") or item.get("category") or "Alert",
            "area": detail.get("area") or "India",
            "severity": detail.get("severity") or "Unknown",
            "urgency": detail.get("urgency") or "",
            "certainty": detail.get("certainty") or "",
            "instruction": detail.get("instruction") or "",
            "expires": detail.get("expires") or "",
            "published": item.get("published") or detail.get("sent") or "",
            "distance": distance,
            "alert_level": detail.get("severity") or "Unknown",
            "source": detail.get("source") or "SACHET/NDMA CAP",
            "match_type": match_type,
            "url": item.get("link", ""),
        })

        if len(matches) >= SACHET_MAX_ALERTS_RETURNED:
            break

    severity_rank = {"extreme": 4, "severe": 3, "moderate": 2, "minor": 1, "unknown": 0}
    matches.sort(
        key=lambda alert: (
            severity_rank.get((alert.get("severity") or "unknown").casefold(), 0),
            alert.get("published", ""),
        ),
        reverse=True,
    )

    return matches, {
        "name": "SACHET/NDMA CAP RSS",
        "status": "ok",
        "url": SACHET_RSS_URL,
        "items_checked": len(feed_items),
        "imd_api_key_configured": bool(IMD_API_KEY),
    }
