"""Download 10 years of NASA POWER monthly rainfall for the three regions (needs internet).

Usage (from the repository root):  python scripts/fetch_nasa_power.py
Writes data/nasa_power_<region>.json, which project4_rainfall.ipynb picks up automatically.

Source: NASA Langley Research Center POWER Project, https://power.larc.nasa.gov
(parameter PRECTOTCORR = bias-corrected precipitation, community AG).
"""
import json
import urllib.request
from pathlib import Path

POINTS = {  # approximate town-centre coordinates (lat, lon)
    "Kampala": (0.3476, 32.5825),
    "Gulu": (2.7724, 32.2881),
    "Mbarara": (-0.6072, 30.6545),
}
START, END = 2014, 2023
URL = ("https://power.larc.nasa.gov/api/temporal/monthly/point?parameters=PRECTOTCORR"
       "&community=AG&latitude={lat}&longitude={lon}&start={start}&end={end}&format=JSON")

if __name__ == "__main__":
    out_dir = Path(__file__).resolve().parents[1] / "data"
    out_dir.mkdir(exist_ok=True)
    for name, (lat, lon) in POINTS.items():
        url = URL.format(lat=lat, lon=lon, start=START, end=END)
        print(f"Fetching {name} … ", end="", flush=True)
        with urllib.request.urlopen(url, timeout=60) as resp:
            payload = json.load(resp)
        (out_dir / f"nasa_power_{name.lower()}.json").write_text(json.dumps(payload))
        print("saved")
