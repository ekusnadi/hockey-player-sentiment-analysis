import requests
import time
import csv
from datetime import datetime, timezone


def fetch_more_comments(post_id, children, headers):
    ids = ",".join(children[:100])
    url = f"https://www.reddit.com/api/morechildren.json?api_type=json&link_id=t3_{post_id}&children={ids}"
    response = requests.get(url, headers=headers)
    time.sleep(1)
    data = response.json()
    return data.get("json", {}).get("data", {}).get("things", [])

def parse_comments(items, post_id, headers, all_comments):
    more_ids = []
    for item in items:
        if item["kind"] == "t1":
            data = item["data"]
            # only save top-level comments
            if data["parent_id"].startswith("t3_"):
                all_comments.append({
                    "body": data["body"],
                    "time_utc": datetime.fromtimestamp(data["created_utc"], tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                })
        elif item["kind"] == "more":
            more_ids.extend(item["data"]["children"])

    for i in range(0, len(more_ids), 100):
        more_items = fetch_more_comments(post_id, more_ids[i:i+100], headers)
        parse_comments(more_items, post_id, headers, all_comments)

def get_all_comments(post_url):
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}
    json_url = post_url.rstrip("/") + ".json?limit=500"

    response = requests.get(json_url, headers=headers)
    time.sleep(1)
    data = response.json()

    post_id = data[0]["data"]["children"][0]["data"]["id"]
    all_comments = []
    parse_comments(data[1]["data"]["children"], post_id, headers, all_comments)

    return all_comments



if __name__ == "__main__":
    url = "https://www.reddit.com/r/SanJoseSharks/comments/1s7rdoc/game_thread_st_louis_blues_313011_san_jose_sharks/"
    comments = get_all_comments(url)
    comments.sort(key=lambda x: x["time_utc"])

    file_name = "comments.csv"

    with open(file_name, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["body", "time_utc"])
        writer.writeheader()
        writer.writerows(comments)

    print(f"Saved {len(comments)} comments to {file_name}")
