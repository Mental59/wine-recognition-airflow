import os
import json

def get_data_dir():
    data_dir = os.getenv('AIRFLOW_DATA_PATH')

    if data_dir is None:
        raise Exception('AIRFLOW_DATA_PATH is None')
    
    return data_dir

def load_data_info():
    data_dir = get_data_dir()
    return json.load(open(os.path.join(data_dir, 'data_info', 'data_info.json')))
