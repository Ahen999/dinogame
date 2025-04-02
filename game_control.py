import json

def get_config_data():
    with open('config.json', 'r') as f:
        data = json.load(f)
    return data
