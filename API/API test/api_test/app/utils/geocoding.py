import requests

def geocode_address(code_postal, ville, quartier=None):
    # Mock: retourne des coordonnées fixes pour test
    return {
        "latitude": 48.8566,
        "longitude": 2.3522
    }

def reverse_geocode(lat, lon):
    """
    Retourne le code postal à partir de la latitude et longitude (via Nominatim).
    """
    try:
        url = f"https://nominatim.openstreetmap.org/reverse?format=json&lat={lat}&lon={lon}&addressdetails=1"
        headers = {"User-Agent": "immo-api/1.0"}
        resp = requests.get(url, headers=headers, timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            return data.get("address", {}).get("postcode", None)
    except Exception as e:
        print("DEBUG reverse_geocode error:", e)
    return None 