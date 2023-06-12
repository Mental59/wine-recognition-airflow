from datetime import datetime

from airflow import DAG
from airflow.operators.python import PythonOperator

from lib.defaults import DEFAULT_ARGS

def generate_freq_dict(params, **kwargs):
    import os
    from lib.data_master import DataGenerator, DataLoader, DataSaver
    from lib.utils import load_data_info, get_data_dir

    os.chdir(get_data_dir())
    conf: dict = kwargs['dag_run'].conf

    data_info = load_data_info()

    byword = params['byword']
    freq_dict = DataGenerator.generate_freq_dict(
        *[DataLoader.load_csv_data(path) for path in conf['data_paths']],
        keys=data_info['keys']['all'],
        byword=byword
    )

    output_folder = conf['output'] + '_byword' if byword else conf['output']
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    
    DataSaver.save_frequency_dictionary(
        frequency_dictionary=freq_dict,
        csv_folder=output_folder,
        excel_path=output_folder + '.xlsx'
    )
        
with DAG(
    dag_id='generate_freq_dict_from_transformed_data',
    default_args=DEFAULT_ARGS,
    description='DAG for generating frequency dictionary from transformed source data',
    start_date=datetime.now(),
    schedule_interval=None,
    catchup=False
) as dag:
    generate_byword_freq_dict_task = PythonOperator(
        task_id='generate_byword_freq_dict',
        provide_context=True,
        params={
            'byword': True
        },
        python_callable=generate_freq_dict
    )
    
    generate_freq_dict_task = PythonOperator(
        task_id='generate_freq_dict',
        provide_context=True,
        params={
            'byword': False
        },
        python_callable=generate_freq_dict
    )
