---
title: Recomendador Semântico de Livros
emoji: 📚
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# 📚 Recomendador Semântico de Livros

Aplicação Dash para recomendar livros a partir de uma descrição livre usando NLP.

## Métodos comparados

1. **Similaridade de cosseno**: compara o embedding da consulta com os embeddings dos livros.
2. **Cross-Encoder Re-Ranking**: recupera candidatos por cosseno e reordena com um Cross-Encoder.

## Variáveis do Space

```text
DATA_REPO_ID=seu_usuario/nome_do_dataset
BASE_FILENAME=base_livros_app.parquet
EMBEDDINGS_FILENAME=embeddings_livros.npy
```

Se o dataset for público, não precisa de token.

## Modelos padrão

```text
BI_ENCODER_MODEL_NAME=sentence-transformers/multi-qa-MiniLM-L6-cos-v1
CROSS_ENCODER_MODEL_NAME=cross-encoder/ms-marco-MiniLM-L6-v2
```
