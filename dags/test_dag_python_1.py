from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

from lib.data_master import test_numpy
from lib.nn import test_torch

default_args = {
    'owner': 'mental',
    'retries': 3,
    'retry_delay': timedelta(seconds=10)
}

def test_imports():
    import torch
    import matplotlib
    import tqdm
    from num2words import num2words
    from TorchCRF import CRF
    import mlflow
    import pandas
    import numpy

    print('torch version:', torch.__version__)
    print('matplotlib version:', matplotlib.__version__)
    print('tqdm version:', tqdm.__version__)
    print('num2words:', num2words(35))
    print('TorchCRF:', CRF(5))
    print('mlflow version:', mlflow.__version__)
    print('numpy version:', numpy.__version__)
    print('pandas version:', pandas.__version__)


with DAG(
    dag_id='test_dag_python_1',
    default_args=default_args,
    description='DAG for testing lib imports',
    start_date=datetime(1, 1, 1),
    schedule_interval=None,
    catchup=False
) as dag:
    # test_imports_task = PythonOperator(
    #     task_id='test_imports',
    #     python_callable=test_imports,
    # )

    # test_imports_task

    test_numpy = PythonOperator(
        task_id = 'test_numpy',
        python_callable=test_numpy
    )

    test_torch = PythonOperator(
        task_id = 'test_torch',
        python_callable=test_torch
    )

    test_numpy >> test_torch
