from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

default_args = {
    'owner': 'mental',
    'retries': 3,
    'retry_delay': timedelta(seconds=10)
}

def task():
    print("Hello Airflow!")

with DAG(
    dag_id='test_dag_python_1',
    default_args=default_args,
    description='Python test DAG',
    start_date=datetime(1, 1, 1),
    schedule_interval=None,
) as dag:
    task1 = PythonOperator(
        task_id='first_task',
        python_callable=task,
    )

    task1
