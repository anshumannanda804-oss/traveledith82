import requests

GDACS_API_URL = "https://www.gdacs.org/gdacsapi/api/events/geteventlist/format/json"

def get_gdacs_alerts(country):
    """
    Fetch alerts from GDACS and format them similarly to SACHET alerts 
    so the frontend can display them uniformly.
    """
    params = {
        "country": country,
        "eventtypes": "EQ,TC,FL,VO,WF,DR",
        "alertlevel": "Green;Orange;Red"
    }

    try:
        response = requests.get(GDACS_API_URL, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        features = data.get("features", [])
        formatted_alerts = []
        
        for feature in features:
            props = feature.get("properties", {})
            event_type = props.get("eventtype", "Alert")
            alert_level = props.get("alertlevel", "Unknown")
            name = props.get("name", "GDACS Event")
            desc = props.get("description", "")
            date = props.get("fromdate", "")
            url = props.get("url", "")
            
            formatted_alerts.append({
                "name": name,
                "event": event_type,
                "area": country,
                "severity": alert_level,
                "urgency": "Expected",
                "certainty": "Observed",
                "instruction": desc,
                "expires": "",
                "published": date,
                "distance": 0,
                "alert_level": alert_level,
                "source": "GDACS",
                "match_type": "country",
                "url": url,
            })
            
        return formatted_alerts, {
            "name": "GDACS API",
            "status": "ok",
            "url": GDACS_API_URL,
            "items_checked": len(features)
        }

    except Exception as e:
        print("GDACS Error:", e)
        return [], {
            "name": "GDACS API",
            "status": "error",
            "message": str(e),
            "url": GDACS_API_URL
        }
