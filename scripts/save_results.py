import json

def save_results(results, file_path):
    with open(file_path, 'w') as f:
        json.dump(results, f)
