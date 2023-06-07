def get_data_dir():
    import os

    data_dir = os.getenv('AIRFLOW_DATA_PATH')
    if data_dir is None:
        raise Exception('AIRFLOW_DATA_PATH is None')
    
    return data_dir
