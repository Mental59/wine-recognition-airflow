# wine-recognition-airflow

## Initialization

```bash
# On Linux, the quick-start needs to know your host user id and needs to have group id set to 0. Otherwise the files created in dags, logs and plugins will be created with root user ownership. You have to make sure to configure them for the docker-compose:
echo -e "AIRFLOW_UID=$(id -u)" > .env

# On all operating systems, you need to run database migrations and create the first user account. To do this, run.
docker compose up airflow-init

# build wine-recognition-airflow image
./build.sh
```

## Running Airflow
```bash
docker compose up
```