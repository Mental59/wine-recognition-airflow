from datetime import datetime

from airflow import DAG
from airflow.operators.python import PythonOperator

from lib.defaults import DEFAULT_ARGS

def run_preparation(ti, **kwargs):
  import os
  from lib.utils import chdir_runs_folder, create_folder

  chdir_runs_folder()
  conf: dict = kwargs['dag_run'].conf
  run_name = conf['run_name']

  now = datetime.today().strftime('%Y-%m-%d-%H-%M-%S')
  run_id = f'{run_name}_{now}'

  ti.xcom_push(key='run_id', value=run_id)
  create_folder(run_id)

  os.chdir(run_id)

  create_folder('models')
  create_folder('artifacts')
  create_folder('dataset')


def generate_dataset(ti, **kwargs):
  import os
  from lib.data_master import DataLoader, DataGenerator, ComplexGeneratorMain, ComplexGeneratorMenu, CfgValue
  from lib.utils import chdir_run_id_folder, get_data_dir

  conf: dict = kwargs['dag_run'].conf

  run_id = ti.xcom_pull(task_ids='preparation', key='run_id')
  chdir_run_id_folder(run_id)

  data_dir = get_data_dir()
  dataset_paths = [os.path.join(data_dir, path) for path in conf['dataset']['dataset_paths']]
  df = DataLoader.load_csv_and_concat(dataset_paths)

  main_generator = ComplexGeneratorMain(conf['dataset']['pattern_main'])
  menu_generators = ComplexGeneratorMenu.load_patterns(conf['dataset']['pattern_samples'])

  n_rows = int(len(df) * conf['dataset']['pattern_percent'])
  with open(os.path.join('dataset', 'dataset.txt'), 'w', encoding='utf-8') as file:
    final_dataset = [
      DataGenerator.generate_data_text_complex(df, main_generator)
    ]

    for menu_generator in menu_generators:
      column_names = [value.column for value in menu_generator.cfg if value.values is None]
      samples = df[column_names].dropna().sample(n_rows)
      final_dataset.append(DataGenerator.generate_data_text_menu(samples, menu_generator))
    
    file.write(''.join(final_dataset))


with DAG(
    dag_id='train_bilstm_crf',
    default_args=DEFAULT_ARGS,
    description='DAG for training BiLSTM-CRF',
    start_date=datetime.now(),
    schedule_interval=None,
    catchup=False
) as dag:
  preparation_task = PythonOperator(
    task_id='preparation',
    provide_context=True,
    python_callable=run_preparation
  )

  generate_dataset_task = PythonOperator(
    task_id='generate_dataset',
    provide_context=True,
    python_callable=generate_dataset
  )

  preparation_task >> generate_dataset_task
