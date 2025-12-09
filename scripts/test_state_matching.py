"""Simple unit script for matching logic in crop_recommendation.app

Run this without starting the server; it imports the matching helper and
verifies expected behavior for several simulated reverse-geocode payloads.
"""
import os, sys, json

# ensure project root is on sys.path so imports work when running from scripts/
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from crop_recommendation.app import match_state_to_crops

samples = [
    # typical Nominatim response for Hyderabad
    ({"state": "Telangana", "country": "India"}, "Telangana"),
    ({"state": "Telangana State", "country": "India"}, "Telangana"),
    ({"state": "Orissa", "country": "India"}, "Odisha"),
    ({"county": "Karnataka", "country": "India"}, "Karnataka"),
    ({"state": "NCT of Delhi", "country": "India"}, "Delhi"),
    ({"state": "Pondicherry", "country": "India"}, "Puducherry"),
]

for addr, expected_display in samples:
    state, crops, matched_key, debug = match_state_to_crops(addr, {
        # minimal mapping used only for tests
        "Telangana": ["Rice"],
        "Odisha": ["Rice"],
        "Karnataka": ["Rice"],
        "Delhi": ["Wheat"],
        "Puducherry": ["Rice"]
    })
    print('Input address:', json.dumps(addr), '\n  Resolved state:', state, '\n  Matched key:', matched_key, '\n  Crops:', crops, '\n  Debug:', debug, '\n')
    # basic sanity check
    assert matched_key is not None, f"Failed to match {addr}"

print('All matching samples passed.')
