# Рекомендательная система на MovieLens

Практичный, interview-ready двухэтапный рекомендатель на MovieLens с implicit feedback:
- генерация кандидатов
- ранжирование
- офлайн-оценка качества
- FastAPI-сервис
- Docker-упаковка

## Постановка задачи

В продакшене рекомендательные системы обычно строятся как multi-stage pipeline:
1. **генерация кандидатов** быстро сужает большой каталог до небольшого релевантного набора
2. **ранжирование** упорядочивает кандидатов по более богатым признакам

В этом репозитории реализован именно такой подход на MovieLens с упором на понятный код и удобное объяснение на собеседовании.

## Почему это важно

Рекомендательные системы напрямую влияют на:
- вовлечение и удержание пользователей
- глубину сессии
- качество контент-дискавери

Даже простой и аккуратно оцененный baseline часто полезнее, чем сложная, но непрозрачная модель.

## Датасет и implicit feedback

- Датасет: MovieLens (`ml-latest-small` для MVP)
- Ожидаемые файлы:
  - `data/ml-latest-small/ratings.csv`
  - `data/ml-latest-small/movies.csv`
- Преобразование в implicit feedback:
  - оставляем взаимодействия с `rating >= min_rating` (по умолчанию `4.0`)
  - присваиваем бинарную метку (`1`) для дальнейшей постановки ranking-задачи

## Архитектура проекта

```text
ratings.csv + movies.csv
        |
        v
[preprocessing]
- implicit conversion
- фильтрация активных пользователей
- time-based split (train/val/test)
        |
        +--------------------------+
        |                          |
        v                          v
[candidate generation]      [feature engineering]
popularity + item-item      user/item/user-item/recency признаки
        |                          |
        +-------------> [CatBoostRanker]
                               |
                               v
                      [top-K рекомендации]
                               |
                               v
                       [FastAPI /recommend]
```

## Структура репозитория

```text
.
├── app/
│   └── main.py
├── data/
│   └── README.md
├── notebooks/
│   ├── EDA_and_modeling.ipynb
│   └── README.md
├── src/
│   ├── candidate_generation.py
│   ├── config.py
│   ├── data.py
│   ├── evaluate.py
│   ├── features.py
│   ├── inference.py
│   ├── ranking.py
│   └── train.py
├── .gitignore
├── Dockerfile
├── requirements.txt
└── README.md
```

## Этап 1. Генерация кандидатов

1. **Popularity baseline**
   - ранжирование фильмов по числу взаимодействий в train
   - фильтрация уже просмотренных пользователем
   - рекомендация top unseen

2. **Item-item similarity baseline**
   - построение user-item implicit матрицы
   - cosine similarity между item-ами
   - агрегация скоринга по истории пользователя
   - backfill popularity-кандидатами при необходимости

## Этап 2. Ранжирование

- Модель: `CatBoostRanker` (`YetiRank`)
- Обучающий датасет ранжирования:
  - кандидаты для пользователей из validation-периода
  - метка `1`, если кандидат встречается в целевых взаимодействиях периода
- Inference:
  - скоринг кандидатов моделью
  - возврат top-N
  - fallback на popularity для cold-start пользователя

## Используемые признаки

- Пользовательские:
  - `user_interaction_count`
  - `user_active_span_days`
  - `user_recency_hours`
- Предметные (item):
  - `item_interaction_count`
  - `item_unique_users`
  - `item_popularity_rank`
  - `item_popularity_pct`
  - `item_recency_hours`
- User-item / candidate:
  - `item_item_score`
  - `candidate_rank`
  - `candidate_source_is_item_item`
  - `user_item_interaction_count`
  - `hours_since_item_last_seen`
  - `hours_since_user_last_seen`
  - `hours_since_user_item_seen`
  - `user_genre_affinity`

## Оценка качества

- Time-based split:
  - train: ранние взаимодействия
  - val: средний период
  - test: самые поздние взаимодействия
- Метрики:
  - Recall@K
  - MAP@K
  - NDCG@K
- Сравниваются:
  - popularity baseline
  - item-item baseline
  - двухэтапный пайплайн (candidate generation + ranker)

## Таблица результатов

Заполняй только после фактического запуска обучения/оценки.

| Модель | Recall@10 | MAP@10 | NDCG@10 | Комментарий |
|---|---:|---:|---:|---|
| Popularity | TODO | TODO | TODO | baseline |
| Item-item | TODO | TODO | TODO | candidate baseline |
| Two-stage | TODO | TODO | TODO | candidate + ranker |

## Как запустить локально

### 1) Окружение и зависимости

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

macOS / Linux:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Подготовить MovieLens

Положи файлы в:

```text
data/ml-latest-small/
├── ratings.csv
└── movies.csv
```

### 3) Обучить пайплайн

```bash
python -m src.train --data-dir data/ml-latest-small --artifacts-dir artifacts --min-rating 4.0 --top-k-candidates 100 --k-metrics 10
```

Артефакты в `artifacts/`:
- `popularity_model.joblib`
- `item_item_model.joblib`
- `ranker_artifacts.joblib`
- `seen_items_map.joblib`
- `train_df.joblib`
- `movies_df.joblib`
- `metrics.json`

## Запуск API

```bash
uvicorn app.main:app --reload
```

## Пример запроса/ответа API

Запрос:

```bash
curl -X POST "http://127.0.0.1:8000/recommend" -H "Content-Type: application/json" -d "{\"user_id\": 1, \"k\": 10}"
```

Форма ответа:

```json
{
  "user_id": 1,
  "k": 10,
  "recommendations": [
    {
      "movie_id": 260,
      "title": "Название фильма",
      "score": 0.0
    }
  ]
}
```

Важно: `score` зависит от обученной модели и имеет смысл только после генерации артефактов обучения.

## Docker

Сборка:

```bash
docker build -t movielens-recsys .
```

Запуск:

```bash
docker run -p 8000:8000 movielens-recsys
```

## Что улучшить дальше

- добавить matrix factorization (например, ALS) в candidate generation
- улучшить negative sampling для ranking stage
- добавить diversity/novelty ограничения в финальный reranking
- добавить трекинг экспериментов и версионирование моделей
- добавить ANN-индекс для ускорения retrieval на больших каталогах

## Текущий статус

- Реализовано: рабочий MVP двухэтапной рекомендательной системы + API
- Не заполнено: реальные итоговые метрики в README
- TODO: запустить обучение и перенести фактические значения из `artifacts/metrics.json` в таблицу результатов
