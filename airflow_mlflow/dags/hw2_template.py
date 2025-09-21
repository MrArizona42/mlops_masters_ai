# Требования
# Nested runs должны быть видны в MLflow UI.
# Обязательно логировать метрики и параметры
# У зарегистрированной модели есть signature и input_example.
# В Registry должна попасть только одна, лучшая модель.
# Не забудьте про нейминги! Это важно!
# Код структурирован: функции, запуск через if __name__ == "__main__":
# В этом ДЗ убедительно просим обойтись sklearn'ом. В следующем ДЗ сможете опять разгуляться.
# Файл называем ivanov_hw2.py
# Вам необходимо реализовать логирование ML-эксперимента с помощью MLFlow,
# который будет состоять из следующих шагов:

import os

import mlflow
import pandas as pd
from mlflow import MlflowClient
from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error as MSE
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

MY_NAME = "Anton"
MY_SURNAME = "Muradov"
EXPERIMENT_NAME = f"{MY_NAME}_{MY_SURNAME}"
PARENT_RUN_NAME = "MrArizona42"
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")


def prepare_data():
    """
    Скачайте любой датасет.
    Сделайте препроцессинг по желанию (доп баллы не предусмотрены).
    Сделайте train/test split(random_state=42).
    """
    import pandas as pd
    from sklearn.datasets import load_diabetes
    from sklearn.preprocessing import StandardScaler

    data = load_diabetes()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    scaler = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
    X_test = pd.DataFrame(scaler.transform(X_test), columns=X.columns)

    return X_train, X_test, y_train, y_test


def train_and_log(name, model, X_train, y_train, X_test, y_test):
    """
    Подключитесь к MLFlow внутри докер сети
    создайте эксперимент с названием Ivanov_I. Не забудьте проверку на уже существующий эксперимент!
    запустите parent run с вашим ником в телеграм в качестве имени
    Обучите 3 любые модели в цикле (child runs) для каждой:Залогируйте метрики согласно решаемой задаче с помощью MLFlow
    Залогируйте параметры модели с помощью MLFlow
    Сгенерируйте signature через infer_signature и залогируйте модель с input_example с помощью MLFlow
    """

    # Обучаем модель
    model.fit(X_train, y_train)

    # Делаем predict
    y_pred = model.predict(X_test)

    # Получаем описание данных
    signature = infer_signature(X_test, y_pred)

    # Сохраняем модель в артифакторе
    model_info = mlflow.sklearn.log_model(
        model,
        artifact_path=name,
        signature=signature,
        input_example=X_train.sample(5),
    )

    # Сохраняем метрики модели
    mlflow.evaluate(
        model_info.model_uri,
        X_test,
        y_test.values,
        model_type="regressor",
        evaluators=["default"],
    )

    params = model.get_params()
    for param_name, param_value in params.items():
        mlflow.log_param(param_name, param_value)


def main():
    """
    Получите одну основную метрику (на ваш выбор) из каждого child run.
    Выберите модель с лучшей метрикой.
    Зарегистрируйте лучшую модель с именем вида LogReg_{Surname}.
    Переведите её версию в стадию Staging.
    В консоли выведите сообщение с названием лучшей модели и её метрикой (logging — наше все).
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    try:
        experiment_id = mlflow.get_experiment_by_name(EXPERIMENT_NAME).experiment_id
    except mlflow.exceptions.MlflowException:
        experiment_id = mlflow.create_experiment(EXPERIMENT_NAME)

    mlflow.set_experiment(experiment_id)

    models = {
        "LinearRegression": LinearRegression(),
        "Tree": DecisionTreeRegressor(),
        "RandomForestRegressor": RandomForestRegressor(),
    }

    with mlflow.start_run(
        name=PARENT_RUN_NAME, experiment_id=experiment_id, description="Parent run"
    ) as parent_run:
        X_train, X_test, y_train, y_test = prepare_data()

        for model_name, model in models.items():
            with mlflow.start_run(
                name=model_name, experiment_id=experiment_id, nested=True
            ) as child_run:
                train_and_log(model_name, model, X_train, y_train, X_test, y_test)


if __name__ == "__main__":
    main()
