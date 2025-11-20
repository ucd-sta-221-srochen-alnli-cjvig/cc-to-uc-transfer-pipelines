import csv
import time
import requests
import pandas as pd

GOOGLE_API_KEY = "NULL"  

ROUTES_URL = "https://routes.googleapis.com/directions/v2:computeRoutes"

HEADERS = {
    "Content-Type": "application/json",
    "X-Goog-Api-Key": GOOGLE_API_KEY,
    "X-Goog-FieldMask": "routes.distanceMeters,routes.duration",
}


def build_body(cc_row, uc_row):
    """Get CC UC Coordinates"""
    return {
        "origin": {
            "location": {
                "latLng": {
                    "latitude": float(cc_row["lat"]),
                    "longitude": float(cc_row["lon"]),
                }
            }
        },
        "destination": {
            "location": {
                "latLng": {
                    "latitude": float(uc_row["lat"]),
                    "longitude": float(uc_row["lon"]),
                }
            }
        },
        "travelMode": "DRIVE",
        "routingPreference": "TRAFFIC_AWARE",
    }


def main():
    coords_df = pd.read_csv("cc_uc_coords_from_list.csv")
    list_df = pd.read_csv("cc_uc_list.csv")

    name_to_group = {}

    if "CC" in list_df.columns:
        for val in list_df["CC"].dropna():
            s = str(val).strip()
            if s:
                name_to_group[s] = "CC"

    if "UC" in list_df.columns:
        for val in list_df["UC"].dropna():
            s = str(val).strip()
            if s:
                name_to_group[s] = "UC"

    def classify(name):
        name = str(name).strip()
        return name_to_group.get(name, "UNKNOWN")

    coords_df["group"] = coords_df["input_name"].apply(classify)

    coords_df = coords_df.dropna(subset=["lat", "lon"])

    cc_df = coords_df[coords_df["group"] == "CC"].copy()
    uc_df = coords_df[coords_df["group"] == "UC"].copy()

    print(f"CC: {len(cc_df)}, UC: {len(uc_df)}")


    if len(cc_df) == 0 or len(uc_df) == 0:
        raise RuntimeError("Empty Pair:check cc_uc_coords_from_list.csv and cc_uc_list.csv"

    out_file = "cc_uc_drive_distances.csv"
    fieldnames = [
        "cc_name",
        "uc_name",
        "cc_lat",
        "cc_lon",
        "uc_lat",
        "uc_lon",
        "distance_meters",
        "duration_seconds",
        "status",
    ]

    with open(out_file, "w", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()

        total = len(cc_df) * len(uc_df)
        count = 0

        for _, cc_row in cc_df.iterrows():
            for _, uc_row in uc_df.iterrows():
                count += 1
                body = build_body(cc_row, uc_row)

                try:
                    resp = requests.post(ROUTES_URL, headers=HEADERS, json=body)
                    resp.raise_for_status()
                    data = resp.json()
                    route = data["routes"][0]
                    dist = route.get("distanceMeters")
                    dur_str = route.get("duration", "0s")
                    dur_sec = int(dur_str.rstrip("s"))
                    status = "OK"
                except requests.exceptions.HTTPError as e:
                    print("HTTPError:", e)
                    try:
                        print("Response text:", resp.text)
                        status = f"HTTP {resp.status_code}: {resp.text}"
                    except Exception:
                        status = f"HTTP_ERROR: {e}"
                    dist = None
                    dur_sec = None
                except Exception as e:
                    dist = None
                    dur_sec = None
                    status = f"ERROR: {e}"

                writer.writerow(
                    {
                        "cc_name": cc_row["input_name"],
                        "uc_name": uc_row["input_name"],
                        "cc_lat": cc_row["lat"],
                        "cc_lon": cc_row["lon"],
                        "uc_lat": uc_row["lat"],
                        "uc_lon": uc_row["lon"],
                        "distance_meters": dist,
                        "duration_seconds": dur_sec,
                        "status": status,
                    }
                )

                if count % 20 == 0 or count == total:
                    print(f" {count}/{total}")

                time.sleep(0.05)

    print(f"Done! Saved to {out_file}")


if __name__ == "__main__":
    main()
