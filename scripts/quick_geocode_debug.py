"""Quick script to reverse-geocode coordinates and run matching logic without starting the Flask server.

Usage:
  python scripts/quick_geocode_debug.py [lat] [lon]

If no coords provided, defaults to Hyderabad (17.3850,78.4867).
"""
import sys, os, json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import requests
try:
    import requests_cache
except Exception:
    requests_cache = None

from crop_recommendation.app import match_state_to_crops

def main():
    lat = float(sys.argv[1]) if len(sys.argv) > 1 else 17.3850
    lon = float(sys.argv[2]) if len(sys.argv) > 2 else 78.4867

    if requests_cache is not None:
        requests_cache.install_cache(os.path.join('Data', 'geocode_cache'), backend='sqlite', expire_after=86400)

    try:
        r = requests.get('https://nominatim.openstreetmap.org/reverse', params={
            'lat': round(lat,3), 'lon': round(lon,3), 'format': 'json', 'zoom': 5, 'addressdetails': 1
        }, headers={'User-Agent':'EPICS-Agent/1.0'}, timeout=10)
        r.raise_for_status()
        payload = r.json()
        cached_flag = bool(getattr(r, 'from_cache', False))
        print('Reverse geocode payload:')
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        print(f"(cached={cached_flag})")
        address = payload.get('address', {})
        mapping_path = os.path.join(os.path.dirname(__file__), '..', 'crop_recommendation', 'state_to_crops.json')
        with open(mapping_path, 'r', encoding='utf-8') as f:
            state_map = json.load(f)
        state, crops, matched_key, debug = match_state_to_crops(address, state_map)
        print('\nMatch results:')
        print('resolved_state=', state)
        print('matched_key=', matched_key)
        print('crops=', crops)
        print('debug=', debug)
    except Exception as e:
        print('Geocode failed:', e)

if __name__ == '__main__':
    main()
