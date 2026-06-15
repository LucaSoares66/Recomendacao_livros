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

Aplicação web desenvolvida com **Dash** para recomendação de livros utilizando **NLP**, embeddings semânticos e similaridade de cosseno.

## Objetivo

Recomendar livros a partir de uma descrição livre digitada pelo usuário, como:

```text
fantasy books with elves, dragons and magic kingdoms
```

O sistema busca livros semanticamente próximos à descrição informada.

## Como funciona

1. Cada livro é representado por um texto com informações como título, autor, categoria e descrição.
2. Esse texto é convertido em embedding pelo modelo `multi-qa-MiniLM-L6-cos-v1`.
3. A descrição do usuário também é transformada em embedding.
4. O sistema calcula a similaridade de cosseno entre a consulta e os livros.
5. Os livros mais semelhantes são retornados como recomendação.

## Modelo utilizado

```text
sentence-transformers/multi-qa-MiniLM-L6-cos-v1
```

O modelo foi escolhido por ser adequado para busca semântica, comparando consultas curtas com documentos candidatos.

## Tecnologias

* Python
* Dash
* Plotly
* Pandas
* NumPy
* Sentence Transformers
* Hugging Face Spaces
* Docker

## Arquivos principais

```text
app.py
Dockerfile
requirements.txt
assets/styles.css
README.md
```

## Dados

Os dados foram preparados a partir da base **Amazon Books Reviews**, disponível no Kaggle.

Arquivos utilizados na aplicação:

```text
base_livros_app.parquet
embeddings_livros_float32.npy
```

## Variáveis de ambiente

```text
DATA_REPO_ID
BASE_FILENAME
EMBEDDINGS_FILENAME
MODEL_NAME
```

## Execução local

```bash
pip install -r requirements.txt
python app.py
```

A aplicação roda em:

```text
http://localhost:7860
```

## Deploy

O projeto foi publicado no **Hugging Face Spaces** usando Docker.

## Autor

Luca Soares
IESB — Campus Sul

