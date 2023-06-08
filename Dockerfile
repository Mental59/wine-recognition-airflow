FROM apache/airflow:2.6.1-python3.10

RUN pip install --user --upgrade pip
RUN pip install --no-cache-dir --user torch torchvision torchaudio matplotlib tqdm num2words TorchCRF mlflow openpyxl xlsxwriter
