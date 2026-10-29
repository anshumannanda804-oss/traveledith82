import os
import re
import requests
import numpy as np
from math import radians, sin, cos, sqrt, atan2
from email.utils import parsedate_to_datetime
from sklearn.neighbors import NearestNeighbors


from currency import convert_currency
from currency import currency_symbol as _currency_symbol

TOMTOM_API_KEY = os.getenv("TOMTOM_API_KEY", "MS39QFw2ocwUvr3eBI6C9sVkiVfjoTYK").strip()
HOTEL_SEARCH_RADIUS_KM = 15

# ---------------- CONSTANTS ----------------
CITIES_WITH_AIRPORTS = {
    # Odisha
    "bhubaneswar", "jharsuguda", "rourkela", "jeypore",
    # Andhra Pradesh / Telangana
    "hyderabad", "visakhapatnam", "vijayawada", "tirupati",
    # Arunachal Pradesh / Assam / Northeast
    "itanagar", "guwahati", "imphal", "agartala", "aizawl", "kohima",
    # Bihar
    "patna",
    # Chhattisgarh
    "raipur",
    # Goa
    "goa", "panaji",
    # Gujarat
    "ahmedabad", "surat", "vadodara", "rajkot",
    # Himachal Pradesh
    "dharamshala", "dharamsala", "kangra", "mcleod ganj", "mcleodganj", "shimla", "kullu", "manali",
    # J&K / Ladakh
    "srinagar", "leh", "jammu", "ladakh",
    # Jharkhand
    "ranchi",
    # Karnataka
    "bangalore", "bengaluru", "mysore", "mangalore", "hubballi",
    # Kerala
    "kochi", "thiruvananthapuram", "kozhikode", "kannur",
    # Madhya Pradesh
    "indore", "bhopal", "jabalpur", "khajuraho",
    # Maharashtra
    "mumbai", "pune", "aurangabad", "nagpur", "nashik",
    # Manipur / Meghalaya
    "imphal", "shillong",
    # Punjab / Haryana / Delhi
    "amritsar", "chandigarh", "ludhiana", "delhi", "new delhi", "gurgaon",
    # Rajasthan
    "jaipur", "jodhpur", "udaipur", "jaisalmer",
    # Sikkim / West Bengal
    "gangtok", "kolkata", "bagdogra", "siliguri", "darjeeling",
    # Tamil Nadu
    "chennai", "coimbatore", "madurai", "tiruchirappalli", "trichy",
    # Uttar Pradesh
    "lucknow", "varanasi", "agra", "prayagraj", "allahabad",
    # Uttarakhand
    "dehradun", "pantnagar",
    # International
    "paris", "london", "dubai", "singapore", "bangkok", "bali", "tokyo",
    "new york", "los angeles", "barcelona", "rome", "sydney", "amsterdam",
    "maldives", "iceland",
    # Added
    "agatti", "bilaspur", "gaya", "kishangarh", "port blair", "tezpur", "vidyanagar",
}

AIRPORT_ACCESS_BY_CITY = {
    # Odisha destinations served by nearby airports
    "puri": "bhubaneswar",
    "konark": "bhubaneswar",
    "chilika lake": "bhubaneswar",
    "sambalpur": "jharsuguda",
    "balangir": "jharsuguda",
    "berhampur": "bhubaneswar",
    # Domestic tourist places where the nearest airport is not the city name
    "bodh gaya": "gaya",
    "rajgir": "patna",
    "bilaspur": "bilaspur",
    "faridabad": "delhi",
    "jamshedpur": "ranchi",
    "coorg": "mangalore",
    "hampi": "vidyanagar",
    "munnar": "kochi",
    "alleppey": "kochi",
    "cherrapunji": "shillong",
    "tawang": "tezpur",
    "ajmer": "kishangarh",
    "mathura": "agra",
    "rishikesh": "dehradun",
    "mussoorie": "dehradun",
    "nainital": "pantnagar",
    "sundarbans": "kolkata",
    "ooty": "coimbatore",
    # Island routes should prefer flights and hide normal bus/train options
    "andaman": "port blair",
    "andaman and nicobar": "port blair",
    "andaman and nicobar islands": "port blair",
    "port blair": "port blair",
    "lakshadweep": "agatti",
    "agatti": "agatti",
}

FLIGHT_ONLY_REGIONS = {
    "andaman",
    "andaman and nicobar",
    "andaman and nicobar islands",
    "lakshadweep",
}

NO_DIRECT_TRAIN_DESTINATIONS = {
    "andaman",
    "andaman and nicobar",
    "andaman and nicobar islands",
    "port blair",
    "lakshadweep",
    "agatti",
    "tawang",
    "kaziranga",
    "cherrapunji",
    "shillong",
    "gangtok",
    "manali",
    "kullu",
    "dharamshala",
    "dharamsala",
    "mcleod ganj",
    "mcleodganj",
    "leh",
    "ladakh",
    "coorg",
    "munnar",
    "mussoorie",
    "nainital",
    "sundarbans",
}

# ---------------- LOCATION & MATH ----------------
def calculate_distance(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

def simplify_match_text(value):
    return re.sub(r"[^a-z0-9]+", " ", (value or "").lower()).strip()

def text_matches_terms(text, terms):
    simple_text = f" {simplify_match_text(text)} "
    for term in terms:
        simple_term = simplify_match_text(term)
        if len(simple_term) < 3:
            continue
        if f" {simple_term} " in simple_text or (
            " " in simple_term and simple_term in simple_text
        ):
            return simple_term
    return None

def is_point_in_india_bounds(lat, lon):
    try:
        lat = float(lat)
        lon = float(lon)
    except (TypeError, ValueError):
        return False
    return 6.0 <= lat <= 38.8 and 68.0 <= lon <= 98.8

def parse_rss_date(value):
    if not value:
        return ""
    try:
        return parsedate_to_datetime(value).isoformat()
    except (TypeError, ValueError, IndexError, AttributeError):
        return value

def normalize_place_name(name):
    return re.sub(r"\s+", " ", (name or "").strip().lower())

def is_flight_only_region(name, state=""):
    candidates = {normalize_place_name(name), normalize_place_name(state)}
    return any(
        region in candidate
        for candidate in candidates
        for region in FLIGHT_ONLY_REGIONS
    )

def has_direct_train_access(name, state=""):
    normalized = normalize_place_name(name)
    if is_flight_only_region(name, state):
        return False
    return normalized not in NO_DIRECT_TRAIN_DESTINATIONS

def has_airport(lat, lon, city_name=None):
    if city_name:
        city = normalize_place_name(city_name)
        if city in CITIES_WITH_AIRPORTS or city in AIRPORT_ACCESS_BY_CITY:
            return True
        known_airport_places = set(CITIES_WITH_AIRPORTS) | set(AIRPORT_ACCESS_BY_CITY)
        for known in known_airport_places:
            if len(city) >= 4 and (known in city or city in known):
                return True

    try:
        geo_url = "https://nominatim.openstreetmap.org/reverse"
        res = requests.get(
            geo_url,
            params={"lat": lat, "lon": lon, "format": "json"},
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=5
        )
        if res.status_code == 200:
            data = res.json()
            address = data.get("address", {})
            candidates = [
                address.get("city", ""),
                address.get("town", ""),
                address.get("village", ""),
                address.get("county", ""),
                address.get("state_district", ""),
                address.get("suburb", ""),
            ]
            for name in candidates:
                if name and name.lower() in CITIES_WITH_AIRPORTS:
                    return True
    except Exception:
        pass

    try:
        tomtom_url = "https://api.tomtom.com/search/2/poiSearch/airport.json"
        params = {
            "lat": lat,
            "lon": lon,
            "radius": 100000,
            "limit": 5,
            "categorySet": "7315",
            "key": TOMTOM_API_KEY
        }
        res = requests.get(tomtom_url, params=params, timeout=5)
        if res.status_code == 200:
            results = res.json().get("results", [])
            airport_keywords = ["airport", "aerodrome", "air base", "airfield", "airstrip"]
            for r in results:
                poi_name = r.get("poi", {}).get("name", "").lower()
                if any(kw in poi_name for kw in airport_keywords):
                    return True
    except Exception:
        pass

    return False

def estimate_transport_cost(
    distance_km,
    is_international,
    same_state=False,
    flight_possible=True,
    source_name=None,
    destination_name=None,
    source_state="",
    destination_state="",
):
    transport = {}
    flight_only_route = (
        is_flight_only_region(source_name, source_state)
        or is_flight_only_region(destination_name, destination_state)
    )

    if not is_international:
        if not flight_only_route:
            transport["bus"] = int(distance_km * 3)

            if (
                has_direct_train_access(source_name, source_state)
                and has_direct_train_access(destination_name, destination_state)
            ):
                transport["train"] = int(distance_km * 2)

        minimum_flight_distance = 100 if flight_only_route or same_state else 150
        if distance_km >= minimum_flight_distance and flight_possible:
            if distance_km < 500:
                flight_rate = 9
            elif distance_km < 1500:
                flight_rate = 7
            else:
                flight_rate = 6
            transport["flight"] = int(distance_km * flight_rate)
    else:
        if distance_km < 3000:
            flight_rate = 15
        elif distance_km < 7000:
            flight_rate = 12
        else:
            flight_rate = 10
        transport["flight"] = int(distance_km * flight_rate)

    return transport

# ---------------- HOTELS ----------------
def add_hotel_details(hotel, distance, currency="INR"):
    print("Currency received:", currency)
    name = hotel.get("name", "Hotel")
    sum_chars = sum(ord(c) for c in name)
    from currency import convert_currency
    from currency import currency_symbol as _currency_symbol

    # Base hotel price in USD
    base_price_usd = 60 + (sum_chars % 15) * 20

    hotel["price_usd"] = base_price_usd

    # Convert to INR (keep an INR equivalent) and to requested currency
    try:
        base_price_inr = convert_currency(base_price_usd, "INR")
    except Exception:
        base_price_inr = base_price_usd

    try:
        converted_price = convert_currency(base_price_usd, currency)
    except Exception:
        converted_price = base_price_inr
    
    hotel["price"] = converted_price
    hotel["price_inr"] = base_price_inr
    hotel["currency"] = currency
    hotel["symbol"] = _currency_symbol(currency)
    print("Hotel:", hotel["name"])
    print("Converted Price:", hotel["price"])
    print("Currency:", hotel["currency"])


    
    hotel["price"] = converted_price
    hotel["price_inr"] = base_price_inr
    hotel["currency"] = currency
    hotel["symbol"] = _currency_symbol(currency)
    hotel["rating"] = round(3.2 + (sum_chars % 18) * 0.1, 1)
    hotel["distance"] = round(distance, 2)
    
    return hotel

def recommend_hotels_knn(hotels_list, target_budget, target_rating, target_distance):
    if not hotels_list:
        return []
    features = []
    for h in hotels_list:
        feat = [h["price"] / 1000.0, h["rating"] * 2.0, h["distance"]]
        features.append(feat)
    X = np.array(features)
    target = np.array([[target_budget / 1000.0, target_rating * 2.0, target_distance]])
    n_neighbors = min(len(hotels_list), 5)
    if n_neighbors > 0:
        knn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
        knn.fit(X)
        distances, indices = knn.kneighbors(target)
        recommended = [hotels_list[idx] for idx in indices[0]]
        return recommended
    return []

def fetch_tomtom_hotels(destination_lat, destination_lon, currency="INR"):
    if not TOMTOM_API_KEY:
        return [], "TomTom API key is not configured."

    try:
        search_params = {
            "lat": destination_lat,
            "lon": destination_lon,
            "radius": HOTEL_SEARCH_RADIUS_KM * 1000,
            "limit": 50,
            "key": TOMTOM_API_KEY,
        }
        response = requests.get(
            "https://api.tomtom.com/search/2/poiSearch/hotel.json",
            params=search_params,
            timeout=10,
        )
        response.raise_for_status()
        results = response.json().get("results", [])

        if not results:
            response = requests.get(
                "https://api.tomtom.com/search/2/search/hotel.json",
                params=search_params,
                timeout=10,
            )
            response.raise_for_status()
            results = response.json().get("results", [])
    except requests.exceptions.RequestException as error:
        status_code = getattr(getattr(error, "response", None), "status_code", None)
        message = (
            f"TomTom hotel search failed (HTTP {status_code})."
            if status_code else "TomTom hotel search could not be reached."
        )
        print(f"TomTom hotel search error: {error}")
        return [], message
    except ValueError as error:
        print(f"TomTom hotel response error: {error}")
        return [], "TomTom returned an invalid hotel-search response."

    hotels = []
    seen = set()
    for result in results:
        poi = result.get("poi") or {}
        position = result.get("position") or {}
        try:
            lat = float(position["lat"])
            lon = float(position["lon"])
        except (KeyError, TypeError, ValueError):
            continue

        distance = calculate_distance(destination_lat, destination_lon, lat, lon)
        if distance > HOTEL_SEARCH_RADIUS_KM:
            continue

        name = (poi.get("name") or "Unnamed Hotel").strip()
        identity = (name.casefold(), round(lat, 6), round(lon, 6))
        if identity in seen:
            continue
        seen.add(identity)

        hotel = {
            "name": name,
            "lat": lat,
            "lon": lon,
            "source": "tomtom",
        }
        hotels.append(add_hotel_details(hotel, distance, currency))

    return hotels, None
