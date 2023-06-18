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

  create_folder(run_id)

  os.chdir(run_id)

  create_folder('models')
  create_folder('artifacts')
  create_folder('dataset')

  ti.xcom_push(key='run_id', value=run_id)


def generate_dataset(ti, **kwargs):
  import os
  import json
  from lib.data_master import DataLoader, DataGenerator, ComplexGeneratorMain, ComplexGeneratorMenu
  from lib.utils import chdir_run_id_folder, get_data_dir, load_data_info, create_folder
  from lib.nn import generate_tag_to_ix, preprocess_raw_data, compute_max_sentence_len
  from sklearn.model_selection import train_test_split

  conf: dict = kwargs['dag_run'].conf

  run_id = ti.xcom_pull(task_ids='preparation', key='run_id')
  chdir_run_id_folder(run_id)

  data_dir = get_data_dir()
  dataset_paths = [os.path.join(data_dir, path) for path in conf['dataset']['dataset_paths']]
  df = DataLoader.load_csv_and_concat(dataset_paths)

  main_generator = ComplexGeneratorMain(conf['dataset']['pattern_main'])
  menu_generators = ComplexGeneratorMenu.load_patterns(conf['dataset']['pattern_samples'])

  n_rows = int(len(df) * conf['dataset']['pattern_percent'])
  final_dataset = [
    DataGenerator.generate_data_text_complex(df, main_generator)
  ]
  for menu_generator in menu_generators:
    column_names = [value.column for value in menu_generator.cfg if value.values is None]
    samples = df[column_names].dropna().sample(n_rows)
    final_dataset.append(DataGenerator.generate_data_text_menu(samples, menu_generator))
  final_dataset = ''.join(final_dataset)
  with open(os.path.join('dataset', 'final_dataset.txt'), 'w', encoding='utf-8') as file:
    file.write(final_dataset)
  
  sents = DataGenerator.generate_sents2(final_dataset.split('\n'))
  train_data, val_data = train_test_split(sents, test_size=conf['dataset']['test_size'])

  print(train_data[0])
  print(val_data[0])

  case_sensitive = conf['dataset']['case_sensitive_vocab']
  use_num2words = conf['dataset']['num2words']
  train_data = list(preprocess_raw_data(train_data, case_sensitive, use_num2words))
  val_data = list(preprocess_raw_data(val_data, case_sensitive, use_num2words))

  data_info = load_data_info()
  tag_to_ix = generate_tag_to_ix(data_info['keys']['all'])

  with open(os.path.join('dataset', 'tag_to_ix.json'), 'w', encoding='utf-8') as file:
    json.dump(tag_to_ix, file)
  
  with open(os.path.join('dataset', 'metadata.json'), 'w', encoding='utf-8') as file:
    json.dump(dict(
      train_length=len(train_data),
      val_length=len(val_data),
      max_train_sent_length=compute_max_sentence_len(train_data),
      max_val_sent_length=compute_max_sentence_len(val_data)
    ), file)

  sents_folder = os.path.join('dataset', 'sents')
  train_sents_folder = os.path.join(sents_folder, 'train')
  val_sents_folder = os.path.join(sents_folder, 'val')
  create_folder(sents_folder)
  create_folder(train_sents_folder)
  create_folder(val_sents_folder)

  def save_dataset(folder: str, data):
    for index, (words, tags) in enumerate(data):
      with open(os.path.join(folder, f'{index}.json'), 'w', encoding='utf-8') as file:
        json.dump(dict(words=words, tags=tags), file)
  
  save_dataset(train_sents_folder, train_data)
  save_dataset(val_sents_folder, val_data)

  print(
    f'sents: length={len(sents)}'
    f'train_data: length={len(train_data)}',
    f'val_data: length={len(val_data)}',
    sep='\n'
  )


def train(ti, **kwargs):
  import os
  import json
  from lib.utils import chdir_run_id_folder, get_data_dir
  from lib.nn import CustomDataset

  conf: dict = kwargs['dag_run'].conf
  run_id = ti.xcom_pull(task_ids='preparation', key='run_id')
  chdir_run_id_folder(run_id)

  with open(os.path.join('dataset', 'tag_to_ix.json'), 'r', encoding='utf-8') as file:
    tag_to_ix = json.load(file)
  with open(os.path.join(get_data_dir(), conf['dataset']['vocab_path']), 'r', encoding='utf-8') as file:
    word_to_ix = json.load(file)
  with open(os.path.join('dataset', 'metadata.json'), 'r', encoding='utf-8') as file:
    metadata = json.load(file)
  
  case_sensitive_vocab = conf['dataset']['case_sensitive_vocab']
  model_conf = conf['model']

  train_data_path = os.path.join('dataset', 'sents', 'train')
  val_data_path = os.path.join('dataset', 'sents', 'val')

  train_dataset = CustomDataset(
    data_path=train_data_path,
    data_length=metadata['train_length'],
    max_sent_length=metadata['max_train_sent_length'],
    tag_to_ix=tag_to_ix,
    word_to_ix=word_to_ix,
    case_sensitive=case_sensitive_vocab,
  )

  val_dataset = CustomDataset(
    data_path=val_data_path,
    data_length=metadata['val_length'],
    max_sent_length=metadata['max_val_sent_length'],
    tag_to_ix=tag_to_ix,
    word_to_ix=word_to_ix,
    case_sensitive=case_sensitive_vocab
  )

  print(train_dataset[0])
  print(val_dataset[0])


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

  train_task = PythonOperator(
    task_id='train',
    provide_context=True,
    python_callable=train
  )

  preparation_task >> generate_dataset_task >> train_task
