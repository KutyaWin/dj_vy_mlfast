# Churn Service

Сервис для обучения и использования модели прогнозирования оттока клиентов (`churn`) на базе FastAPI и scikit-learn.

## Цель сервиса

Проект решает две основные задачи:

- обучает churn-модель на CSV-датасете;
- отдаёт предсказания, статус модели, схему признаков, историю обучений и health-check через HTTP API.

Сервис поддерживает:

- единый формат ошибок `code / message / details`;
- сохранение обученной модели на диск;
- сохранение истории обучений и метрик;
- endpoint для проверки состояния сервиса;
- запуск локально и в Docker.

## Структура проекта

Текущая структура проекта intentionally оставлена простой и понятной:

```text
main.py                 FastAPI entrypoint, routes, exception handlers
src/models.py           Pydantic-схемы запросов и ответов
src/utils.py            Логика загрузки датасета, обучения, предсказаний и persistence
data/churn_dataset.csv  Исходный датасет churn
models/                 Сохранённые артефакты модели и history
tests/                  Unit и integration тесты
Dockerfile              Контейнеризация сервиса
requirements.txt        Python зависимости
```

### Ответственности модулей

- **`main.py`**
  - API layer
  - endpoint-ы
  - OpenAPI examples
  - global exception handlers
  - логирование верхнего уровня

- **`src/models.py`**
  - request/response схемы
  - error response схемы
  - health / metrics / training history модели

- **`src/utils.py`**
  - загрузка и валидация churn датасета
  - подготовка признаков
  - обучение модели
  - предсказание churn
  - сохранение модели и истории обучений
  - health-check helper

## Формат датасета `churn_dataset.csv`

Файл должен содержать заголовок с колонками:

```text
monthly_fee,usage_hours,support_requests,account_age_months,failed_payments,region,device_type,payment_method,autopay_enabled,churn
```

### Признаки

- **`monthly_fee`** — числовой признак, `float`
- **`usage_hours`** — числовой признак, `float`
- **`support_requests`** — числовой признак, `int`
- **`account_age_months`** — числовой признак, `int`
- **`failed_payments`** — числовой признак, `int`
- **`region`** — категориальный признак, `str`
- **`device_type`** — категориальный признак, `str`
- **`payment_method`** — категориальный признак, `str`
- **`autopay_enabled`** — числовой бинарный признак, `int`
- **`churn`** — target, `0` или `1`

### Пример строк датасета

```csv
monthly_fee,usage_hours,support_requests,account_age_months,failed_payments,region,device_type,payment_method,autopay_enabled,churn
9.99,27.92,1,14,1,america,desktop,card,1,1
19.99,21.48,2,1,0,america,mobile,card,1,0
```

## Основные endpoint-ы

- **`GET /`**
  - базовая проверка доступности API

- **`GET /health`**
  - состояние сервиса
  - показывает доступность датасета и модели

- **`GET /dataset/info`**
  - информация о датасете

- **`GET /dataset/preview`**
  - preview строк датасета

- **`GET /dataset/split-info`**
  - параметры и распределения train/test split

- **`POST /model/train`**
  - обучение churn модели

- **`GET /model/status`**
  - статус и метрики последней обученной модели

- **`GET /model/metrics`**
  - история метрик обучения

- **`GET /model/schema`**
  - ожидаемые признаки и их типы

- **`POST /predict`**
  - предсказание churn для одного клиента или батча клиентов

- **`GET /docs`**
  - Swagger UI

## Локальный запуск

### 1. Создать виртуальное окружение

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Установить зависимости

```bash
python3 -m pip install -r requirements.txt
```

### 3. Запустить сервис

```bash
uvicorn main:app --reload
```

После запуска сервис будет доступен по адресу:

```text
http://127.0.0.1:8000
```

Swagger UI:

```text
http://127.0.0.1:8000/docs
```

Health-check:

```text
http://127.0.0.1:8000/health
```

## Запуск в Docker

### Сборка образа

```bash
docker build -t churn-service .
```

### Запуск контейнера

```bash
docker run --rm -p 8000:8000 churn-service
```

После запуска внутри контейнера должны быть доступны:

- **`/health`**
- **`/docs`**

## Примеры запросов

### Обучение модели: `POST /model/train`

#### Пример 1. Logistic Regression

```bash
curl -X POST http://127.0.0.1:8000/model/train \
  -H "Content-Type: application/json" \
  -d '{
    "model_type": "logreg",
    "hyperparameters": {}
  }'
```

#### Пример 2. Random Forest

```bash
curl -X POST http://127.0.0.1:8000/model/train \
  -H "Content-Type: application/json" \
  -d '{
    "model_type": "random_forest",
    "hyperparameters": {
      "n_estimators": 200,
      "max_depth": 8,
      "min_samples_split": 4
    }
  }'
```

### Предсказание: `POST /predict`

#### Один клиент

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "monthly_fee": 39.99,
    "usage_hours": 87.5,
    "support_requests": 1,
    "account_age_months": 24,
    "failed_payments": 0,
    "region": "North",
    "device_type": "Mobile",
    "payment_method": "Card",
    "autopay_enabled": 1
  }'
```

#### Батч клиентов

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '[
    {
      "monthly_fee": 39.99,
      "usage_hours": 87.5,
      "support_requests": 1,
      "account_age_months": 24,
      "failed_payments": 0,
      "region": "North",
      "device_type": "Mobile",
      "payment_method": "Card",
      "autopay_enabled": 1
    },
    {
      "monthly_fee": 79.99,
      "usage_hours": 12.0,
      "support_requests": 6,
      "account_age_months": 3,
      "failed_payments": 2,
      "region": "West",
      "device_type": "Desktop",
      "payment_method": "Bank Transfer",
      "autopay_enabled": 0
    }
  ]'
```

## Формат ошибок

Сервис возвращает структурированный JSON-ответ:

```json
{
  "code": "model_not_trained",
  "message": "Churn model is not trained. Train the model via POST /model/train first.",
  "details": null
}
```

Типовые коды ошибок:

- **`invalid_feature_type`**
- **`invalid_feature_count`**
- **`empty_prediction_request`**
- **`model_not_trained`**
- **`dataset_not_found`**
- **`dataset_empty`**
- **`unsupported_model_type`**
- **`training_failed`**
- **`internal_server_error`**

## Модель и история обучений

Сервис сохраняет артефакты в директорию `models/`:

- **`churn_model.joblib`** — обученная sklearn pipeline-модель
- **`churn_model_metadata.json`** — метаданные последнего обучения
- **`churn_training_history.json`** — история обучений и метрик

История обучений содержит:

- timestamp
- model type
- model name
- hyperparameters
- accuracy
- f1
- roc_auc

## Тестирование

В проекте есть:

- **unit-like тесты** для data prep и training helpers
- **integration тесты** через `TestClient`

Запуск тестов:

```bash
python3 -m pytest -q
```

Если `pytest` ещё не установлен в окружении, сначала установите зависимости из `requirements.txt`.

## Наблюдаемость

Сервис логирует ключевые события:

- загрузку churn датасета
- запуск и завершение обучения
- вызовы `/predict`
- загрузку модели
- ошибки работы сервиса

## Итог

Проект готов для:

- локального запуска
- обучения churn модели
- выдачи предсказаний
- мониторинга через `/health`
- контейнеризации через Docker
- дальнейшего расширения API и ML-логики
