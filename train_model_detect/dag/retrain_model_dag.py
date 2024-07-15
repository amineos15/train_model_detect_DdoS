from datetime import datetime, timedelta
from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'retrain_model_dag',
    default_args=default_args,
    description='A DAG to retrain prediction model every 15 days',
    schedule_interval=timedelta(days=15),
    start_date=datetime(2023, 1, 1),
    catchup=False,
)

retrain_model_task = DockerOperator(
    task_id='retrain_model',
    image='model-trainer',
    api_version='auto',
    auto_remove=True,
    environment={
        'AWS_ACCESS_KEY_ID': '',
        'AWS_SECRET_ACCESS_KEY': '',
        'AWS_REGION': ''
    },
    docker_url='unix://var/run/docker.sock',
    network_mode='bridge',
    dag=dag,
)

retrain_model_task
