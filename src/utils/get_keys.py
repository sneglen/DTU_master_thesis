import json

def get_keys(path):
    with open(path) as f:
        return json.load(f)
