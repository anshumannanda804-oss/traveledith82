import time
import requests

API_URL = "https://open.er-api.com/v6/latest/USD"

# Cache exchange rates to avoid hammering the API (1 call per hotel was killing performance)
_rates_cache = {"rates": {}, "timestamp": 0}
_CACHE_TTL = 600  # 10 minutes

def get_exchange_rates():
    """
    Fetch latest exchange rates using USD as base currency.
    Results are cached for 10 minutes to avoid repeated API calls.
    """
    now = time.time()
    if _rates_cache["rates"] and (now - _rates_cache["timestamp"]) < _CACHE_TTL:
        return _rates_cache["rates"]

    try:
        response = requests.get(API_URL, timeout=10)
        data = response.json()

        if data["result"] == "success":
            _rates_cache["rates"] = data["rates"]
            _rates_cache["timestamp"] = now
            return data["rates"]
        else:
            return _rates_cache["rates"] or {}

    except Exception as e:
        print("Currency API Error:", e)
        return _rates_cache["rates"] or {}
def convert_currency(amount, target_currency, from_currency="USD"):
    """
    Convert amount from from_currency to target_currency.
    """

    rates = get_exchange_rates()

    if target_currency not in rates or from_currency not in rates:
        return None

    # Convert to USD first (base), then to target
    amount_in_usd = amount / rates[from_currency]
    converted_amount = amount_in_usd * rates[target_currency]

    return round(converted_amount, 2)
def currency_symbol(currency):

    symbols = {
        "USD": "$",
        "INR": "₹",
        "EUR": "€",
        "GBP": "£",
        "JPY": "¥",
        "AUD": "A$",
        "CAD": "C$",
        "SGD": "S$",
        "AED": "د.إ"
    }

    return symbols.get(currency, currency)


if __name__ == "__main__":

    amount = 150

    currency = "INR"

    converted = convert_currency(amount, currency)

    symbol = currency_symbol(currency)

    print(f"${amount} = {symbol}{converted}")    