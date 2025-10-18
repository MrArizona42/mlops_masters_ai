# Описание задания
# Вам необходимо реализовать DAG в Airflow, который осуществляет логирование ML-эксперимента с помощью MLFlow, который будет состоять из следующих шагов:

# 1 init
# Запустить пайплайн, зафиксировать timestamp начала.
# Логировать сообщение о старте.

# 2 data_collection
# Считать данные (любые из интернетов, не качайте там петабайты только ).
# Сохранить сырые данные в бакет S3.
# Залогировать сообщение об успешном сборе данных.

# 3 split_and_preprocess
# Разделить данные на train/test (train_test_split(random_state=42)).
# Сделать минимальный препроцессинг (например, заполнение пропусков, ohe/categorical encoding для пары признаков).
# Сохранить train/test в S3 бакет.
# Залогировать сообщение об успешном препроцессинге.

# 4 train_and_log_mlflow
# Подключитесь к MLFlow внутри AirFlow
# создайте эксперимент с названием Ivanov_Final. Не забудьте проверку на уже существующий эксперимент!
# запустите parent run с вашим ником в телеграм в качестве имени

# Обучите 3 любые модели в цикле (child runs) для каждой:
# Залогируйте метрики согласно решаемой задаче с помощью MLFlow
# Залогируйте параметры модели с помощью MLFlow
# Сгенерируйте signature через infer_signature и залогируйте модель с input_example с помощью MLFlow

# Получите одну основную метрику (на ваш выбор) из каждого child run.
# Выберите модель с лучшей метрикой.
# Зарегистрируйте лучшую модель с именем вида LogReg_{Surname}.
# Переведите её версию в стадию Staging.
# Передайте в следующий шаг run_id лучше модели

# 5 serve model
# Примите по XСom Run ID лучшей модели и засервите ее любым способом (пройденном в курсе PythonOperator/PythonEnvOperator/BashOperator/DocerOperator или из вашей практики)
# В текстовом поле при сдаче ДЗ объясните почему выбрали именно такой спосо

# 6 cleanup (опционально, доп баллы не предусмотрены)

# Требования:
# Nested runs должны быть видны в MLflow UI.
# У зарегистрированной модели есть signature и input_example.
# В Registry должна попасть только одна, лучшая модель.
# Не забудьте про нейминги! Это важно!
# Файл называем ivanov_hw3.py

import io
import logging
from datetime import datetime

import pandas as pd
from airflow import DAG
from airflow.models import Variable
from airflow.operators.python import PythonOperator
from airflow.providers.amazon.aws.hooks.s3 import S3Hook

# -----------------
# Конфигурация переменных
# -----------------
AWS_CONN_ID = "S3_CONNECTION"
MY_NAME = "Anton"
MY_SURNAME = "Muradov"
MLFLOW_EXPERIMENT_NAME = f"{MY_SURNAME}{MY_NAME[0]}_Final"
PARENT_RUN_NAME = "MrArizona42"

S3_BUCKET = Variable.get("S3_BUCKET")
MLFLOW_TRACKING_URI = Variable.get("MLFLOW_TRACKING_URI")
AWS_ACCESS_KEY_ID = Variable.get("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = Variable.get("AWS_SECRET_ACCESS_KEY")

S3_PREFIX = f"{MY_SURNAME}/hw3"
S3_KEY_RAW = f"{S3_PREFIX}/data/raw_tips.csv"
S3_KEY_TRAIN = f"{S3_PREFIX}/data/train.csv"
S3_KEY_TEST = f"{S3_PREFIX}/data/test.csv"
TARGET_COLUMN = "tip"
DATA_SOURCE_URL = (
    "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv"
)


# -----------------
# Утилиты: работа с S3 через BytesIO
# -----------------
def s3_read_csv(hook: S3Hook, bucket: str, key: str) -> pd.DataFrame:
    buf = io.BytesIO()
    hook.get_conn().download_fileobj(bucket, key, buf)
    buf.seek(0)
    return pd.read_csv(buf)


def s3_write_csv(hook: S3Hook, df: pd.DataFrame, bucket: str, key: str) -> None:
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    hook.get_conn().upload_fileobj(buf, bucket, key)


# -----------------
# Таски
# -----------------
def init_pipeline(**context):
    ts = datetime.utcnow().isoformat()
    logging.info(f"pipeline_start={ts}")
    context["ti"].xcom_push(key="pipeline_start", value=ts)


def collect_data(**context):
    hook = S3Hook(aws_conn_id=AWS_CONN_ID)
    logging.info("Downloading raw dataset from internet source")
    df = pd.read_csv(DATA_SOURCE_URL)
    df.drop_duplicates(inplace=True)
    logging.info(f"Collected dataset shape={df.shape}")
    s3_write_csv(hook=hook, df=df, bucket=S3_BUCKET, key=S3_KEY_RAW)
    logging.info("Raw dataset saved to S3")
    context["ti"].xcom_push(key="raw_rows", value=int(df.shape[0]))


def split_and_preprocess(**context):
    from sklearn.model_selection import train_test_split

    hook = S3Hook(aws_conn_id=AWS_CONN_ID)
    df = s3_read_csv(hook=hook, bucket=S3_BUCKET, key=S3_KEY_RAW)
    logging.info(f"Loaded raw data from S3 with shape={df.shape}")

    df = df.dropna().reset_index(drop=True)
    features = df.drop(columns=[TARGET_COLUMN])
    target = df[TARGET_COLUMN]

    features = pd.get_dummies(features, drop_first=True)
    logging.info(f"Features transformed to shape={features.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42
    )

    train_df = X_train.copy()
    train_df[TARGET_COLUMN] = y_train
    test_df = X_test.copy()
    test_df[TARGET_COLUMN] = y_test

    s3_write_csv(hook=hook, df=train_df, bucket=S3_BUCKET, key=S3_KEY_TRAIN)
    s3_write_csv(hook=hook, df=test_df, bucket=S3_BUCKET, key=S3_KEY_TEST)
    logging.info("Train and test datasets stored in S3")

    context["ti"].xcom_push(key="train_rows", value=int(train_df.shape[0]))
    context["ti"].xcom_push(key="test_rows", value=int(test_df.shape[0]))


def train_and_log_mlflow(**context):
    import mlflow
    from mlflow import MlflowClient
    from mlflow.models import infer_signature
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    hook = S3Hook(aws_conn_id=AWS_CONN_ID)
    train_df = s3_read_csv(hook=hook, bucket=S3_BUCKET, key=S3_KEY_TRAIN)
    test_df = s3_read_csv(hook=hook, bucket=S3_BUCKET, key=S3_KEY_TEST)

    X_train = train_df.drop(columns=[TARGET_COLUMN])
    y_train = train_df[TARGET_COLUMN]
    X_test = test_df.drop(columns=[TARGET_COLUMN])
    y_test = test_df[TARGET_COLUMN]

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()
    experiment = client.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
    if experiment is None:
        experiment_id = client.create_experiment(MLFLOW_EXPERIMENT_NAME)
    else:
        experiment_id = experiment.experiment_id

    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    models = {
        "LinearRegression": LinearRegression(),
        "RandomForestRegressor": RandomForestRegressor(
            n_estimators=150, random_state=42
        ),
        "GradientBoostingRegressor": GradientBoostingRegressor(random_state=42),
    }

    best_metric = None
    best_model_name = None
    best_run_id = None
    best_model_uri = None

    with mlflow.start_run(run_name=PARENT_RUN_NAME, experiment_id=experiment_id):
        mlflow.log_param("train_rows", int(X_train.shape[0]))
        mlflow.log_param("test_rows", int(X_test.shape[0]))

        for model_name, model in models.items():
            logging.info(f"Training model {model_name}")
            with mlflow.start_run(run_name=model_name, nested=True) as child_run:
                model.fit(X_train, y_train)
                predictions = model.predict(X_test)
                rmse = float(mean_squared_error(y_test, predictions, squared=False))
                mae = float(mean_absolute_error(y_test, predictions))
                mlflow.log_metric("rmse", rmse)
                mlflow.log_metric("mae", mae)
                mlflow.log_params(model.get_params())

                signature = infer_signature(X_train, model.predict(X_train))
                input_example = X_train.head(5)
                model_info = mlflow.sklearn.log_model(
                    sk_model=model,
                    artifact_path="model",
                    signature=signature,
                    input_example=input_example,
                )

                logging.info(
                    "Model %s logged with rmse=%.4f mae=%.4f run_id=%s",
                    model_name,
                    rmse,
                    mae,
                    child_run.info.run_id,
                )

                if best_metric is None or rmse < best_metric:
                    best_metric = rmse
                    best_model_name = model_name
                    best_run_id = child_run.info.run_id
                    best_model_uri = model_info.model_uri

    registered_model_name = f"{best_model_name}_{MY_SURNAME}"
    try:
        client.create_registered_model(registered_model_name)
    except mlflow.exceptions.MlflowException:
        logging.info("Registered model %s already exists", registered_model_name)

    existing_versions = client.search_model_versions(
        filter_string=f"name='{registered_model_name}'"
    )
    for version in existing_versions:
        try:
            client.delete_model_version_tag(
                name=registered_model_name,
                version=version.version,
                key="latest",
            )
        except mlflow.exceptions.MlflowException:
            logging.debug(
                "No 'latest' tag to remove for %s version %s",
                registered_model_name,
                version.version,
            )

    model_version = client.create_model_version(
        name=registered_model_name,
        source=best_model_uri,
        run_id=best_run_id,
    )
    client.set_model_version_tag(
        name=registered_model_name,
        version=model_version.version,
        key="latest",
        value="true",
    )

    model_registry_url = f"{MLFLOW_TRACKING_URI.rstrip('/')}/#/models/{registered_model_name}/versions/{model_version.version}"

    logging.info(
        "Best model %s with rmse=%.4f registered as %s version %s and tagged as latest; UI: %s",
        best_model_name,
        best_metric,
        registered_model_name,
        model_version.version,
        model_registry_url,
    )

    context["ti"].xcom_push(key="best_model_registry_url", value=model_registry_url)
    context["ti"].xcom_push(key="registered_model_name", value=registered_model_name)

    context["ti"].xcom_push(key="best_run_id", value=best_run_id)
    context["ti"].xcom_push(key="best_model_name", value=best_model_name)
    context["ti"].xcom_push(key="best_model_metric", value=best_metric)


def serve_model(**context):
    import mlflow
    from mlflow import MlflowClient
    from mlflow.pyfunc import load_model

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    ti = context["ti"]
    registered_model_name = ti.xcom_pull(
        key="registered_model_name", task_ids="train_and_log_mlflow"
    )

    hook = S3Hook(aws_conn_id=AWS_CONN_ID)
    test_df = s3_read_csv(hook=hook, bucket=S3_BUCKET, key=S3_KEY_TEST)
    features = test_df.drop(columns=[TARGET_COLUMN])

    client = MlflowClient()
    latest_version = client.search_model_versions(
        filter_string=f"name='{registered_model_name}' and tags.latest='true'"
    )[0]

    best_run_id = latest_version.run_id
    model_uri = f"runs:/{best_run_id}/model"
    model = load_model(model_uri)

    sample = features.head(3)
    predictions = model.predict(sample)
    logging.info("Sample predictions for serving step: %s", predictions.tolist())
    logging.info(
        "Serving performed via mlflow.pyfunc inside PythonOperator for lightweight in-process inference"
    )


default_args = {"owner": f"{MY_NAME} {MY_SURNAME}", "retries": 1}


with DAG(
    dag_id="hw33",
    default_args=default_args,
    start_date=datetime(2025, 1, 1),
    schedule_interval="@daily",
    catchup=False,
    tags=["mlops"],
) as dag:
    init_pipeline = PythonOperator(
        task_id="init_pipeline", python_callable=init_pipeline
    )
    collect_data = PythonOperator(task_id="collect_data", python_callable=collect_data)
    split_and_preprocess = PythonOperator(
        task_id="split_and_preprocess", python_callable=split_and_preprocess
    )
    train_and_log_mlflow = PythonOperator(
        task_id="train_and_log_mlflow",
        python_callable=train_and_log_mlflow,
    )
    serve_model = PythonOperator(
        task_id="serve_model",
        python_callable=serve_model,
    )  # Можете заменить на любой другой оператор!

    (
        init_pipeline
        >> collect_data
        >> split_and_preprocess
        >> train_and_log_mlflow
        >> serve_model
    )
