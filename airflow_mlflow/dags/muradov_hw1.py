# DAG должен быть написан в отдельном .py файле.
# Использовать PythonOperator.
# Все шаги реализовать как отдельные Python-функции.
# Использовать logging для логирования происходящего в шагах. Что именно логировать - на ваш вкус.
# Собирать метрики пайплайна с помощью XCom.
# Файл model_metrics.json — метрики модели.
# Файл pipeline_metrics.json — технические метрики пайплайна (время старта/окончания, длительность обучения).
# Оба файла должны сохраняться в S3.
# Все данные, модели и метрики должны сохраняться в S3 по адресу {bucket_name}/{your_surname}.
# Доступ к S3 через S3Hook. Имя AWS connection строго - S3_CONNECTION. Имя вашего бакета берется строго из Variable S3_BUCKET.
# В этом ДЗ это совсем не нужно, но если очень хочется - можно использовать дополнительные библиотеки - тогда на помощь придет PythonEnvOperator.
# Ваш файл должен называться ivanov_hw1.py
# owner DAG'a - ваши Фамилия и Имя!


import io
import logging
import os
import tempfile
from datetime import datetime

import joblib
import pandas as pd
import requests
from airflow import DAG
from airflow.models import Variable
from airflow.operators.python import PythonOperator
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# -----------------
# Конфигурация переменных
# -----------------
AWS_CONN_ID = "S3_CONNECTION"
S3_BUCKET = Variable.get("S3_BUCKET")
MY_NAME = "Anton"
MY_SURNAME = "Muradov"

S3_KEY_MODEL_METRICS = f"{MY_SURNAME}/model_metrics.json"
S3_KEY_PIPELINE_METRICS = f"{MY_SURNAME}/pipeline_metrics.json"
S3_KEY_MODEL = f"{MY_SURNAME}/hw1/model.pkl"


# -----------------
# Утилиты: работа с S3 через BytesIO
# -----------------
def s3_read_csv(hook: S3Hook, bucket: str, key: str) -> pd.DataFrame:
    """Read CSV from S3 with error handling and proper resource management."""
    try:
        buf = io.BytesIO()
        hook.get_conn().download_fileobj(bucket, key, buf)
        buf.seek(0)
        df = pd.read_csv(buf, encoding="utf-8")
        logging.info(f"Successfully read CSV from s3://{bucket}/{key}")
        return df
    except Exception as e:
        logging.error(f"Failed to read CSV from s3://{bucket}/{key}: {e}")
        raise


def s3_write_csv(hook: S3Hook, df: pd.DataFrame, bucket: str, key: str) -> None:
    """Write CSV to S3 with error handling."""
    try:
        buf = io.BytesIO()
        df.to_csv(buf, index=False, encoding="utf-8")
        buf.seek(0)
        hook.get_conn().upload_fileobj(buf, bucket, key)
        logging.info(f"Successfully uploaded CSV to s3://{bucket}/{key}")
    except Exception as e:
        logging.error(f"Failed to write CSV to s3://{bucket}/{key}: {e}")
        raise


# -----------------
# Таски
# -----------------


def init_pipeline(**context):
    """
    Запустить пайплайн, зафиксировать timestamp начала.
    Логировать сообщение о старте.
    """
    start_ts = datetime.utcnow().isoformat()
    logging.info(f"Запуск пайплайна: {start_ts}")
    context["ti"].xcom_push(key="pipeline_start", value=start_ts)


def collect_data(**context):
    """
    Считать данные (любые из интернетов, не качайте там петабайты только ).
    Сохранить сырые данные в бакет S3.
    Залогировать сообщение об успешном сборе данных.
    """
    import pandas as pd
    from sklearn.datasets import load_diabetes

    diabetes_data = load_diabetes()
    data = pd.DataFrame(diabetes_data.data, columns=diabetes_data.feature_names)
    data["target"] = diabetes_data.target
    logging.info(f"Loaded data shape: {data.shape}")

    hook = S3Hook(aws_conn_id=AWS_CONN_ID)
    s3_write_csv(
        hook=hook, df=data, bucket=S3_BUCKET, key=f"{MY_SURNAME}/hw1/data/raw_data.csv"
    )
    logging.info("Данные успешно сохранены в S3.")


def split_and_preprocess(**context):
    """
    Разделить данные на train/test (train_test_split(random_state=42)).
    Сделать минимальный препроцессинг (например, заполнение пропусков, ohe/categorical encoding для пары признаков).
    Сохранить train/test в S3 бакет.
    Залогировать сообщение об успешном препроцессинге.
    """
    from sklearn.model_selection import train_test_split

    hook = S3Hook(aws_conn_id=AWS_CONN_ID)
    df = s3_read_csv(
        hook=hook, bucket=S3_BUCKET, key=f"{MY_SURNAME}/hw1/data/raw_data.csv"
    )

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    logging.info(f"Train data shape: {train_df.shape}")
    logging.info(f"Test data shape: {test_df.shape}")

    s3_write_csv(
        hook=hook,
        df=train_df,
        bucket=S3_BUCKET,
        key=f"{MY_SURNAME}/hw1/data/train_data.csv",
    )
    s3_write_csv(
        hook=hook,
        df=test_df,
        bucket=S3_BUCKET,
        key=f"{MY_SURNAME}/hw1/data/test_data.csv",
    )

    logging.info("Данные успешно разделены и сохранены в S3.")


def train_model(**context):
    """
    Обучить простую модель (например, RandomForest).
    Зафиксировать время начала и конца обучения.
    Сохранить модель в S3.
    Залогировать сообщение об успешном завершении обучения.
    """

    hook = S3Hook(aws_conn_id=AWS_CONN_ID)
    train_df = s3_read_csv(
        hook=hook, bucket=S3_BUCKET, key=f"{MY_SURNAME}/hw1/data/train_data.csv"
    )

    X_train = train_df.drop(columns=["target"])
    y_train = train_df["target"]

    model = LinearRegression()
    start_time = datetime.utcnow()
    model.fit(X_train, y_train)
    end_time = datetime.utcnow()

    training_duration = (end_time - start_time).total_seconds()
    logging.info(f"Model training completed in {training_duration} seconds.")

    # Save model to S3
    model_buf = io.BytesIO()
    joblib.dump(model, model_buf)
    model_buf.seek(0)

    hook = S3Hook(aws_conn_id=AWS_CONN_ID)
    hook.get_conn().upload_fileobj(model_buf, S3_BUCKET, S3_KEY_MODEL)
    logging.info(f"Model successfully saved to s3://{S3_BUCKET}/{S3_KEY_MODEL}")

    context["ti"].xcom_push(key="training_start", value=start_time.isoformat())
    context["ti"].xcom_push(key="training_end", value=end_time.isoformat())
    context["ti"].xcom_push(key="training_duration", value=training_duration)


def collect_metrics_model(**context):
    """
    Посчитать метрики качества (например, accuracy, f1-score).
    Сохранить файл model_metrics.json в S3.
    """
    import json

    hook = S3Hook(aws_conn_id=AWS_CONN_ID)
    test_df = s3_read_csv(
        hook=hook, bucket=S3_BUCKET, key=f"{MY_SURNAME}/hw1/data/test_data.csv"
    )
    X_test = test_df.drop(columns=["target"])
    y_test = test_df["target"]
    logging.info(f"Test data successfully loaded")

    model_buf = io.BytesIO()
    hook.get_conn().download_fileobj(S3_BUCKET, S3_KEY_MODEL, model_buf)
    model_buf.seek(0)
    model = joblib.load(model_buf)
    logging.info(f"Model successfully loaded from s3://{S3_BUCKET}/{S3_KEY_MODEL}")

    y_pred = model.predict(X_test)

    metrics = {
        "mse": mean_squared_error(y_test, y_pred),
        "r2_score": r2_score(y_test, y_pred),
    }

    logging.info(f"Model metrics: {metrics}")

    metrics_json = json.dumps(metrics)
    metrics_buf = io.BytesIO(metrics_json.encode("utf-8"))
    metrics_buf.seek(0)

    hook.get_conn().upload_fileobj(metrics_buf, S3_BUCKET, S3_KEY_MODEL_METRICS)
    logging.info(
        f"Model metrics successfully saved to s3://{S3_BUCKET}/{S3_KEY_MODEL_METRICS}"
    )


def collect_metrics_pipeline(**context):
    """
    Сохранить метрики всего пайплайна (timestamp начала и конца, время обучения) в файл pipeline_metrics.json и выгрузить в S3.
    Залогировать сообщение об успешном завершении.
    """
    import json

    ti = context["ti"]
    pipeline_start = ti.xcom_pull(key="pipeline_start", task_ids="init_pipeline")
    training_start = ti.xcom_pull(key="training_start", task_ids="train_model")
    training_end = ti.xcom_pull(key="training_end", task_ids="train_model")
    training_duration = ti.xcom_pull(key="training_duration", task_ids="train_model")
    pipeline_end = datetime.utcnow().isoformat()

    pipeline_metrics = {
        "pipeline_start": pipeline_start,
        "pipeline_end": pipeline_end,
        "training_start": training_start,
        "training_end": training_end,
        "training_duration_seconds": training_duration,
    }

    logging.info(f"Pipeline metrics: {pipeline_metrics}")

    metrics_json = json.dumps(pipeline_metrics)
    metrics_buf = io.BytesIO(metrics_json.encode("utf-8"))
    metrics_buf.seek(0)

    hook = S3Hook(aws_conn_id=AWS_CONN_ID)
    hook.get_conn().upload_fileobj(metrics_buf, S3_BUCKET, S3_KEY_PIPELINE_METRICS)
    logging.info(
        f"Pipeline metrics successfully saved to s3://{S3_BUCKET}/{S3_KEY_PIPELINE_METRICS}"
    )


def cleanup(**context):
    """
    Удалить временные файлы/ train-test split в S3.
    """


default_args = {"owner": f"{MY_NAME} {MY_SURNAME}", "retries": 1}

with DAG(
    dag_id="hw1",
    default_args=default_args,
    start_date=datetime(2025, 1, 1),
    schedule_interval="@daily",
    catchup=False,
    tags=["mlops"],
) as dag:

    t1 = PythonOperator(task_id="init_pipeline", python_callable=init_pipeline)
    t2 = PythonOperator(task_id="collect_data", python_callable=collect_data)
    t3 = PythonOperator(
        task_id="split_and_preprocess", python_callable=split_and_preprocess
    )
    t4 = PythonOperator(task_id="train_model", python_callable=train_model)
    t5 = PythonOperator(
        task_id="collect_metrics_model", python_callable=collect_metrics_model
    )
    t6 = PythonOperator(
        task_id="collect_metrics_pipeline", python_callable=collect_metrics_pipeline
    )
    t7 = PythonOperator(task_id="cleanup", python_callable=cleanup)

    t1 >> t2 >> t3 >> t4 >> t5 >> t6 >> t7
