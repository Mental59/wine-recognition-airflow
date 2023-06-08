from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

default_args = {
    'owner': 'mental',
    'retries': 0,
    'retry_delay': timedelta(seconds=10)
}

def generate_csv(params, **kwargs):
    import os
    from lib.utils import get_data_dir
    from lib.data_master import DataLoader

    data_dir = get_data_dir()
    conf: dict = kwargs['dag_run'].conf

    data = DataLoader.load_excel_data(os.path.join(data_dir, conf['source_data_path']), fillna=True)
    only_completed_rows = params['only_completed_rows']
    drop_not_add_columns = params['drop_not_add_columns']

    data = DataLoader.preprocess(
        data=data,
        fill_bottle_size=conf.get('fill_bottle_size', 750.0),
        only_completed_rows=only_completed_rows,
        drop_not_add_columns=drop_not_add_columns
    )

    path, ext = os.path.splitext(conf['output_data_path'])
    path += '_only_completed_rows' if only_completed_rows else '_all_rows'
    path += '_drop_not_add' if drop_not_add_columns else '_all_columns'

    data.to_csv(os.path.join(data_dir, path + ext))
    
with DAG(
    dag_id='generate_csv_from_source_excel',
    default_args=default_args,
    description='DAG for generating transformed csv file from source excel file',
    start_date=datetime.now(),
    schedule_interval=None,
    catchup=False
) as dag:
    generate_csv_task1 = PythonOperator(
        task_id='generate_csv_only_completed_rows_drop_not_add',
        provide_context=True,
        params={
            'only_completed_rows': True,
            'drop_not_add_columns': True
        },
        python_callable=generate_csv,
        )
    
    generate_csv_task2 = PythonOperator(
        task_id='generate_csv_all_rows_drop_not_add',
        provide_context=True,
        params={
            'only_completed_rows': False,
            'drop_not_add_columns': True
        },
        python_callable=generate_csv,
        )
    
    generate_csv_task3 = PythonOperator(
        task_id='generate_csv_only_completed_rows_all_columns',
        provide_context=True,
        params={
            'only_completed_rows': True,
            'drop_not_add_columns': False
        },
        python_callable=generate_csv,
        )
    
    generate_csv_task4 = PythonOperator(
        task_id='generate_csv_all_rows_all_columns',
        provide_context=True,
        params={
            'only_completed_rows': False,
            'drop_not_add_columns': False
        },
        python_callable=generate_csv,
        )
