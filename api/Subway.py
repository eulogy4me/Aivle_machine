import pandas as pd
import requests
import time
import os

api_key = ""
path = os.getcwd() + "/data"
df = pd.read_csv(path + "/data.csv")

results = {}

def keyword_search(lat, lon, keyword, radius):
    url = "https://dapi.kakao.com/v2/local/search/keyword.json"
    headers = {"Authorization": f"KakaoAK {api_key}"}
    params = {
        "query": keyword,
        "x": lon,
        "y": lat,
        "radius": radius,
        "size": 15
    }
    response = requests.get(url, headers=headers, params=params, timeout=5)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Keyword Search Error: {response.status_code}, {response.text}")
    return None

for index, row in df.iterrows():
    lat, lon, name = row["Latitude"], row["Longitude"], row["단지명"]
    print(f"Processing Latitude={lat}, Longitude={lon}")

    data = keyword_search(lat, lon, keyword="지하철", radius=500)
    if data:
        
        unique_key = (name, lat, lon)
        if unique_key not in results:
            results[unique_key] = {
                "Latitude": lat,
                "Longitude": lon,
                "Places": set()
            }
        
        for place in data.get("documents", []):
            place_data = (
                place.get("place_name"),
                f"{place.get('distance')}m"
            )
            results[unique_key]["Places"].add(place_data)

    time.sleep(0.5)

processed_results = []
for unique_key, info in results.items():
    name, lat, lon = unique_key
    processed_results.append({
        "Place": name,
        "Latitude": lat,
        "Longitude": lon,
        "Places": list(info["Places"])
    })

results_df = pd.DataFrame(processed_results)

results_df.to_csv(path + "/results.csv", index=False, encoding="utf-8-sig")
