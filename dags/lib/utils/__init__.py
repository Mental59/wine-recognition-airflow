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

def create_folder(path: str):
    if not os.path.exists(path):
        os.mkdir(path)

def chdir_runs_folder():
    os.chdir(os.path.join(get_data_dir(), 'runs'))

def chdir_run_id_folder(run_id: str):
    chdir_runs_folder()
    os.chdir(run_id)

def split_price(price: str):
    price = price.replace(',', '')
    divided_price = str(int(float(price) / 5))
    return divided_price + '/' + str(int(float(price))) 

def load_pattern(pattern_name: str):
    return json.load(
        open(os.path.join(get_data_dir(), 'patterns', pattern_name))
    )
