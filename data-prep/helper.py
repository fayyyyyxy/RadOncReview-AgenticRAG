import json
import os

def save_json(disease, suffix, save_dir, json_data):
    os.makedirs(save_dir, exist_ok=True)
    data_path = os.path.join(save_dir, f"{disease}_{suffix}.json")
    with open(data_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=4, ensure_ascii=False)

def load_json(disease, suffix, save_dir):
    data_path = os.path.join(save_dir, f"{disease}_{suffix}.json")
    with open(data_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)
    return json_data