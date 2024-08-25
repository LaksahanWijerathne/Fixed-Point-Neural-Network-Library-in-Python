# data_processing.py

def normalize_data(data, min_val, max_val):
    return [(x - min_val) / (max_val - min_val) for x in data]
