import requests

# Free Exchange Rate API
API_URL = "https://open.er-api.com/v6/latest/USD"


def fetch_exchange_rates():
    """
    Fetch latest exchange rates.
    Base Currency = USD
    """

    try:
        response = requests.get(API_URL, timeout=10)
        response.raise_for_status()

        data = response.json()

        if data["result"] == "success":
            return data["rates"]

        return None

    except requests.exceptions.RequestException as e:
        print("API Error:", e)
        return None