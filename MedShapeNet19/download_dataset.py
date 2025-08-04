import os
import json
import argparse
import re
import urllib.request
import csv

def get_filename(cd):
    match = re.search(r'filename="([^"]+\.stl)"', cd, re.IGNORECASE)
    return match.group(1) if match else None
    
def download_files(class_data, class_folder, split_path):
    os.makedirs(class_folder, exist_ok=True)
    split_info = []
    for sample in class_data["urls"][:2]:
        url = sample["url"]
        with urllib.request.urlopen(url) as response:
            content_disposition = response.headers.get('Content-Disposition')
            if content_disposition:
                filename = get_filename(content_disposition)
            else:
                print(f"no content disposition for {url}")
        split = sample["split"]
        if not filename:
            print(f"no filename for {url}")
        file_path = os.path.join(class_folder, filename)
        try:
            print(f"Downloading {url} to {file_path}...")
            urllib.request.urlretrieve(url, file_path)
            split_info.append([file_path, split])
        except Exception as e:
            print(f"Failed to download {url}: {e}")
    with open(split_path, "a", newline='') as file:
        csv.writer(file).writerows(split_info)

def main():
    parser = argparse.ArgumentParser(description="Download dataset files from JSON description.")
    parser.add_argument("json_file", help="Path to the JSON file")
    parser.add_argument("dataset_folder", help="Path to the output dataset folder")
    parser.add_argument("--class", dest="class_name", help="Optional: specific class to download")

    args = parser.parse_args()

    with open(args.json_file, "r") as f:
        data = json.load(f)

    if "data" not in data or not isinstance(data["data"], list):
        print("Error: JSON does not contain a valid 'data' list.")
        return

    found = False
    split_path = os.path.join(args.dataset_folder, "split_info.csv")
    for entry in data["data"]:
        class_name = entry.get("class")
        if args.class_name:
            if class_name == args.class_name:
                class_folder = os.path.join(args.dataset_folder, class_name)
                download_files(entry, class_folder, split_path)
                found = True
                break
        else:
            class_folder = os.path.join(args.dataset_folder, class_name)
            download_files(entry, class_folder, split_path)
            found = True

    if args.class_name and not found:
        print(f"Error: Class '{args.class_name}' not found in JSON.")

if __name__ == "__main__":
    main()