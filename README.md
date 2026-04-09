# MovieLens Two-Stage Recommender System

[Русская версия](#ru) | [English version](#en)

## RU

### TL;DR
Двухэтапная рекомендательная система на MovieLens: candidate generation + ranking + offline evaluation + API.

### Гипотезы
1. Two-stage архитектура лучше по latency/quality.
2. Ranker улучшает релевантность top-K.
3. Offline-валидация ускоряет безопасные итерации.

### Метрики
`Precision@K`, `Recall@K`, `MAP@K`, `NDCG@K`, latency.

## EN

### Overview
An educational two-stage recommender on MovieLens with candidate generation, ranking, and offline evaluation.
