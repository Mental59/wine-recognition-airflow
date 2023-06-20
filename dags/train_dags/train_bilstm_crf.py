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
  # menu_generators = ComplexGeneratorMenu.load_patterns(conf['dataset']['pattern_samples'])

  final_dataset = [
    DataGenerator.generate_data_text_complex(df, main_generator)
  ]

  # n_rows = int(len(df) * conf['dataset']['pattern_percent'])
  # for menu_generator in menu_generators:
  #   column_names = [value.column for value in menu_generator.cfg if value.values is None]
  #   samples = df[column_names].dropna().sample(n_rows)
  #   final_dataset.append(DataGenerator.generate_data_text_menu(samples, menu_generator))
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
  import numpy as np
  import torch
  from torch.utils.data import DataLoader
  from torch.optim import Adam
  from torch.optim.lr_scheduler import ReduceLROnPlateau
  import sklearn.metrics as metrics
  from lib.utils import chdir_run_id_folder, get_data_dir
  from lib.nn import CustomDataset, BiLSTM_CRF, train, get_model_confidence, plot_losses
  from lib.data_master import count_unk_foreach_tag, DataAnalyzer

  conf: dict = kwargs['dag_run'].conf
  case_sensitive_vocab = conf['dataset']['case_sensitive_vocab']
  vocab_path = conf['dataset']['vocab_path']
  device = conf['model']['device']
  batch_size = conf['model']['batch_size']
  embedding_dim = conf['model']['embedding_dim']
  hidden_dim = conf['model']['hidden_dim']
  num_epochs = conf['model']['num_epochs']
  learning_rate = conf['model']['learning_rate']
  scheduler_factor = conf['model']['scheduler_factor']
  scheduler_patience = conf['model']['scheduler_patience']
  weight_decay = conf['model']['weight_decay']

  run_id = ti.xcom_pull(task_ids='preparation', key='run_id')
  chdir_run_id_folder(run_id)

  with open(os.path.join('dataset', 'tag_to_ix.json'), 'r', encoding='utf-8') as file:
    tag_to_ix = json.load(file)
  with open(os.path.join(get_data_dir(), vocab_path), 'r', encoding='utf-8') as file:
    word_to_ix = json.load(file)
  with open(os.path.join('dataset', 'metadata.json'), 'r', encoding='utf-8') as file:
    metadata = json.load(file)
  
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

  dataloaders = {
    'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True),
    'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
  }

  model_conf = dict(
    vocab_size=len(word_to_ix),
    num_tags=len(tag_to_ix),
    embedding_dim=embedding_dim,
    hidden_dim=hidden_dim,
    padding_idx=word_to_ix['PAD']
  )

  model = BiLSTM_CRF(**model_conf).to(device)
  optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
  scheduler = ReduceLROnPlateau(optimizer, factor=scheduler_factor, patience=scheduler_patience)

  model, losses = train(
    model=model,
    optimizer=optimizer,
    dataloaders=dataloaders,
    device=device,
    num_epochs=num_epochs,
    output_dir='models',
    scheduler=scheduler,
    verbose=True
  )

  with open(os.path.join('models', 'tag_to_ix.json'), 'w', encoding='utf-8') as file:
    json.dump(tag_to_ix, file)
  with open(os.path.join('models', 'word_to_ix.json'), 'w', encoding='utf-8') as file:
    json.dump(word_to_ix, file)
  with open(os.path.join('models', 'model_conf.json'), 'w', encoding='utf-8') as file:
    json.dump(model_conf, file)

  val_dataset_raw_data = list(val_dataset.raw_data())
  y_val_true = [tags for _, tags in val_dataset_raw_data]

  y_val_pred = []
  tags = list(tag_to_ix.keys())
  model.eval()
  with torch.no_grad():
    for x_batch, y_batch, mask_batch, _ in dataloaders['val']:
        x_batch, mask_batch = x_batch.to(device), mask_batch.to(device)
        y_batch_pred = model(x_batch, mask_batch)
        y_val_pred.extend(y_batch_pred)
  y_val_pred = [[tags[tag] for tag in sentence] for sentence in y_val_pred]

  X_test = [
    torch.tensor(val_dataset.sentence_to_indices(sentence), dtype=torch.int64) for sentence, _ in val_dataset_raw_data
  ]

  unk_foreach_tag = count_unk_foreach_tag(X_test, y_val_true, list(tag_to_ix), val_dataset.word_to_ix[val_dataset.unk])
  model_confidence = np.mean(get_model_confidence(model, X_test, device))

  test_eval = [list(zip(sentence, tags, y_val_pred[index])) for index, (sentence, tags) in enumerate(val_dataset_raw_data)]

  y_val_true_flat = [item for sublist in y_val_true for item in sublist]
  y_val_pred_flat = [item for sublist in y_val_pred for item in sublist]
  results = dict(
    conf=conf,
    model_confidence=model_confidence,
    unk_foreach_tag=unk_foreach_tag,
    metrics={
      'f1-score': metrics.f1_score(y_val_true_flat, y_val_pred_flat, average='weighted', labels=tags),
      'precision': metrics.precision_score(y_val_true_flat, y_val_pred_flat, average='weighted', labels=tags),
      'recall': metrics.recall_score(y_val_true_flat, y_val_pred_flat, average='weighted', labels=tags),
      'accuracy': metrics.accuracy_score(y_val_true_flat, y_val_pred_flat)
    },
  )
  with open(os.path.join('artifacts', 'results.json'), 'w', encoding='utf-8') as file:
    json.dump(results, file)

  with open(os.path.join('artifacts', 'classification_report.txt'), 'w', encoding='utf-8') as file:
    file.write(metrics.classification_report(y_val_true_flat, y_val_pred_flat, labels=tags, digits=3))
  
  DataAnalyzer.analyze(
    test_eval=test_eval,
    keys=list(tag_to_ix),
    table_save_path=os.path.join('artifacts', 'colored-table.xlsx'),
    diagram_save_path=os.path.join('artifacts', 'diagram.png')
  )

  plot_losses(
    losses=losses,
    figsize=(12, 8),
    show=False,
    savepath=os.path.join('artifacts', 'losses.png')
  )


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
