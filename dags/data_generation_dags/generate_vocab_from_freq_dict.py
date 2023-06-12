from datetime import datetime

from airflow import DAG
from airflow.operators.python import PythonOperator

from lib.defaults import DEFAULT_ARGS

def generate_vocab(params, **kwargs):
  import os
  import re
  import string
  import json

  from tqdm import tqdm

  from lib.utils import get_data_dir, load_data_info
  from lib.data_master import DataLoader

  os.chdir(get_data_dir())
  conf: dict = kwargs['dag_run'].conf
  data_info = load_data_info()

  regex = re.compile(r'(^[%s]*)|([%s]*$)' % (re.escape(string.punctuation), re.escape(string.punctuation)))
  num_words = data_info['num_words']
  additional_words = conf['additional_words']

  for v in conf['dictionary_paths_vocab_names']:
    freq_dict_path = v['freq_dict_path']
    output = v['output']

    freq_dict = DataLoader.load_frequency_dictionary(freq_dict_path)
    vocab = {}

    i = 0
    for key in tqdm(freq_dict):
        for word in filter(lambda x: len(x.split()) == 1, freq_dict[key].value.values):
            word = regex.sub('', word)
            if word not in vocab:
                vocab[word] = i
                i += 1

    for p in string.punctuation:
        if p not in vocab:
            vocab[p] = i
            i += 1
    
    for w in additional_words:
        vocab[w] = i
        i += 1

    with open(output, 'w', encoding='utf-8') as file:
       json.dump(vocab, file)

    print('Vocab length:', len(vocab))
    
    for w in num_words:
       if w not in vocab:
          vocab[w] = i
          i += 1
    
    output_path, output_ext = os.path.splitext(output)

    with open(f'{output_path}_numwords{output_ext}', 'w', encoding='utf-8') as file:
       json.dump(vocab, file)

    print('Vocab length:', len(vocab)) 
      

with DAG(
    dag_id='generate_vocab_from_freq_dict',
    default_args=DEFAULT_ARGS,
    description='DAG for generating "word-index" vocabulary for neural network training',
    start_date=datetime.now(),
    schedule_interval=None,
    catchup=False
) as dag:
  task1 = PythonOperator(
     task_id='generate_vocab',
     provide_context=True,
     python_callable=generate_vocab
  )
