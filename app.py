import os
from pathlib import Path
from functools import lru_cache

import numpy as np
import pandas as pd
import plotly.express as px

from dash import Dash, html, dcc, dash_table, Input, Output, State
from huggingface_hub import hf_hub_download
from sentence_transformers import SentenceTransformer, CrossEncoder


DATA_REPO_ID = os.getenv("DATA_REPO_ID", "").strip()
BASE_FILENAME = os.getenv("BASE_FILENAME", "base_livros_app.parquet")
EMBEDDINGS_FILENAME = os.getenv("EMBEDDINGS_FILENAME", "embeddings_livros.npy")

BI_ENCODER_MODEL_NAME = os.getenv(
    "BI_ENCODER_MODEL_NAME",
    "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
)
CROSS_ENCODER_MODEL_NAME = os.getenv(
    "CROSS_ENCODER_MODEL_NAME",
    "cross-encoder/ms-marco-MiniLM-L6-v2"
)
HF_TOKEN = os.getenv("HF_TOKEN", None)

EXEMPLOS_BUSCA = [
    "fantasy books with elves, dragons, magic kingdoms and epic adventures",
    "mystery thriller with detective, murder investigation and suspense",
    "romantic drama set during war with emotional story",
    "books about artificial intelligence, technology and society",
    "historical books about world war and human suffering",
    "spiritual books about meditation, awareness and inner peace",
]


def extrair_categoria(cat):
    if pd.isna(cat):
        return "Sem categoria"
    cat = str(cat).replace("[", "").replace("]", "").replace("'", "").replace('"', "")
    primeira = cat.split(",")[0].strip()
    return primeira if primeira else "Sem categoria"


def formatar_autores(autores):
    if pd.isna(autores):
        return "Autor não informado"
    autores = str(autores).replace("[", "").replace("]", "").replace("'", "").replace('"', "")
    return autores if autores.strip() else "Autor não informado"


def montar_texto_documento(base):
    if "texto_final" in base.columns:
        texto = base["texto_final"].fillna("").astype(str)
    else:
        colunas_texto = [
            c for c in ["Title", "description", "authors", "categories", "resumo_reviews", "texto_reviews"]
            if c in base.columns
        ]
        if not colunas_texto:
            texto = base["Title"].fillna("").astype(str)
        else:
            texto = base[colunas_texto].fillna("").astype(str).agg(" ".join, axis=1)
    return texto.str.slice(0, 1200)


@lru_cache(maxsize=1)
def carregar_bi_encoder():
    return SentenceTransformer(BI_ENCODER_MODEL_NAME)


@lru_cache(maxsize=1)
def carregar_cross_encoder():
    return CrossEncoder(CROSS_ENCODER_MODEL_NAME)


@lru_cache(maxsize=1)
def carregar_dados():
    base_path = Path(BASE_FILENAME)
    emb_path = Path(EMBEDDINGS_FILENAME)

    if not base_path.exists() or not emb_path.exists():
        if DATA_REPO_ID:
            base_path = Path(hf_hub_download(
                repo_id=DATA_REPO_ID,
                filename=BASE_FILENAME,
                repo_type="dataset",
                token=HF_TOKEN
            ))
            emb_path = Path(hf_hub_download(
                repo_id=DATA_REPO_ID,
                filename=EMBEDDINGS_FILENAME,
                repo_type="dataset",
                token=HF_TOKEN
            ))
        else:
            raise FileNotFoundError(
                "Arquivos de dados não encontrados. Configure DATA_REPO_ID ou suba os arquivos no Space."
            )

    base = pd.read_parquet(base_path)
    embeddings = np.load(emb_path).astype("float32")

    normas = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.clip(normas, 1e-12, None)

    if "Title" not in base.columns:
        raise ValueError("A base precisa ter uma coluna chamada 'Title'.")

    for col, default in [
        ("authors", "Autor não informado"),
        ("categories", "Sem categoria"),
        ("nota_media", np.nan),
        ("qtd_reviews", np.nan),
        ("ratingsCount", np.nan),
    ]:
        if col not in base.columns:
            base[col] = default

    base["categoria_principal"] = base["categories"].apply(extrair_categoria)
    base["authors_fmt"] = base["authors"].apply(formatar_autores)
    base["texto_documento"] = montar_texto_documento(base)

    n = min(len(base), len(embeddings))
    return base.iloc[:n].reset_index(drop=True), embeddings[:n]


def recomendar_cosseno(descricao, top_k=10):
    modelo = carregar_bi_encoder()
    base, embeddings = carregar_dados()
    embedding_consulta = modelo.encode([descricao], normalize_embeddings=True).astype("float32")[0]
    similaridades = embeddings @ embedding_consulta
    resultados = base.copy()
    resultados["score_cosseno"] = similaridades
    return resultados.sort_values("score_cosseno", ascending=False).head(top_k).copy()


def recomendar_cross_encoder(descricao, top_k=10, candidatos_iniciais=50):
    modelo = carregar_bi_encoder()
    reranker = carregar_cross_encoder()
    base, embeddings = carregar_dados()

    embedding_consulta = modelo.encode([descricao], normalize_embeddings=True).astype("float32")[0]
    similaridades = embeddings @ embedding_consulta

    candidatos = base.copy()
    candidatos["score_cosseno"] = similaridades
    candidatos = candidatos.sort_values("score_cosseno", ascending=False).head(candidatos_iniciais).copy()

    pares = [[descricao, texto] for texto in candidatos["texto_documento"].fillna("").astype(str).tolist()]
    candidatos["score_cross_encoder"] = reranker.predict(pares, batch_size=16, show_progress_bar=False)

    return candidatos.sort_values("score_cross_encoder", ascending=False).head(top_k).copy()


def gerar_cards(df, metodo):
    cards = []
    score_col = "score_cosseno" if metodo == "cosseno" else "score_cross_encoder"
    score_label = "Cosseno" if metodo == "cosseno" else "Cross-Encoder"
    for pos, (_, row) in enumerate(df.iterrows(), start=1):
        nota = row.get("nota_media", np.nan)
        qtd = row.get("qtd_reviews", np.nan)
        score = row.get(score_col, np.nan)
        nota_txt = "Sem nota" if pd.isna(nota) else f"Nota média: {nota:.2f}"
        qtd_txt = "Reviews não informadas" if pd.isna(qtd) else f"{int(qtd)} reviews"
        score_txt = "Score indisponível" if pd.isna(score) else f"{score_label}: {score:.3f}"
        cards.append(html.Div(className="book-card", children=[
            html.Div(className="rank-badge", children=f"#{pos}"),
            html.Div(className="book-info", children=[
                html.H3(row.get("Title", "Título não informado")),
                html.P(row.get("authors_fmt", "Autor não informado"), className="book-author"),
                html.Div(row.get("categoria_principal", "Sem categoria"), className="category-pill"),
                html.Div(className="book-meta", children=[
                    html.Span(nota_txt), html.Span(qtd_txt), html.Span(score_txt)
                ])
            ])
        ]))
    return cards


def gerar_grafico(df, metodo):
    if df.empty:
        return px.bar(title="Sem resultados")
    score_col = "score_cosseno" if metodo == "cosseno" else "score_cross_encoder"
    titulo = "Top livros por similaridade de cosseno" if metodo == "cosseno" else "Top livros após Cross-Encoder"
    eixo = "Similaridade de cosseno" if metodo == "cosseno" else "Score Cross-Encoder"
    grafico_df = df.copy()
    grafico_df["titulo_curto"] = grafico_df["Title"].astype(str).str.slice(0, 42)
    fig = px.bar(
        grafico_df.sort_values(score_col, ascending=True),
        x=score_col,
        y="titulo_curto",
        orientation="h",
        title=titulo,
        labels={score_col: eixo, "titulo_curto": "Livro"},
        hover_data=["Title", "categoria_principal"]
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=70, b=30),
        height=410
    )
    return fig


def preparar_tabela(df, metodo):
    score_col = "score_cosseno" if metodo == "cosseno" else "score_cross_encoder"
    score_nome = "Score cosseno" if metodo == "cosseno" else "Score Cross-Encoder"
    colunas = ["Title", "authors_fmt", "categoria_principal", "nota_media", "qtd_reviews", "ratingsCount", score_col]
    colunas = [c for c in colunas if c in df.columns]
    out = df[colunas].copy().rename(columns={
        "Title": "Título",
        "authors_fmt": "Autores",
        "categoria_principal": "Categoria",
        "nota_media": "Nota média",
        "qtd_reviews": "Qtd. reviews",
        "ratingsCount": "Ratings count",
        score_col: score_nome,
    })
    if "Nota média" in out.columns:
        out["Nota média"] = out["Nota média"].round(2)
    if score_nome in out.columns:
        out[score_nome] = out[score_nome].round(4)
    return out


def criar_data_table(df):
    return dash_table.DataTable(
        data=df.to_dict("records"),
        columns=[{"name": c, "id": c} for c in df.columns],
        page_size=10,
        style_table={"overflowX": "auto"},
        style_cell={
            "backgroundColor": "#111827", "color": "#E5E7EB", "border": "1px solid #374151",
            "fontFamily": "Inter, Arial, sans-serif", "fontSize": "14px", "padding": "10px",
            "textAlign": "left", "whiteSpace": "normal", "height": "auto",
        },
        style_header={"backgroundColor": "#1F2937", "fontWeight": "bold", "color": "#FFFFFF"},
        style_data_conditional=[{"if": {"row_index": "odd"}, "backgroundColor": "#0F172A"}],
    )


app = Dash(__name__, title="Recomendador de Livros com NLP", suppress_callback_exceptions=True)
server = app.server


def bloco_status():
    try:
        base, embeddings = carregar_dados()
        return html.Div(className="status-card", children=[
            html.Div("Base carregada com sucesso", className="status-title"),
            html.Div(f"{len(base):,} livros | {embeddings.shape[1]} dimensões por embedding".replace(",", "."), className="status-subtitle")
        ])
    except Exception as e:
        return html.Div(className="status-card status-error", children=[
            html.Div("Dados ainda não configurados", className="status-title"),
            html.Div(str(e), className="status-subtitle")
        ])


app.layout = html.Div(className="page", children=[
    html.Div(className="hero", children=[
        html.Div(className="hero-content", children=[
            html.Div("📚 NLP + Deep Learning", className="eyebrow"),
            html.H1("Recomendador Semântico de Livros"),
            html.P("Compare dois métodos: similaridade de cosseno e reordenação avançada com Cross-Encoder.")
        ]),
        html.Div(className="hero-panel", children=[
            html.Div("Modelos", className="panel-label"),
            html.Div(f"Bi-Encoder: {BI_ENCODER_MODEL_NAME}", className="model-name"),
            html.Div(f"Cross-Encoder: {CROSS_ENCODER_MODEL_NAME}", className="model-name"),
            bloco_status()
        ])
    ]),
    html.Div(className="main-grid", children=[
        html.Div(className="search-card", children=[
            html.H2("Descreva o livro desejado"),
            html.P("O app retorna duas listas: uma por cosseno e outra por Cross-Encoder.", className="hint"),
            dcc.Textarea(id="query-input", value=EXEMPLOS_BUSCA[0], className="query-box"),
            html.Div(className="controls-row", children=[
                html.Div(className="slider-wrap", children=[
                    html.Label("Quantidade de recomendações"),
                    dcc.Slider(id="top-k", min=5, max=20, step=1, value=10, marks={5:"5",10:"10",15:"15",20:"20"})
                ]),
                html.Div(className="slider-wrap", children=[
                    html.Label("Candidatos iniciais para o Cross-Encoder"),
                    dcc.Slider(id="candidate-k", min=20, max=100, step=10, value=50, marks={20:"20",50:"50",100:"100"})
                ])
            ]),
            html.Button("Comparar recomendações", id="search-button", className="primary-button"),
            html.Div(className="examples", children=[
                html.Div("Exemplos de busca", className="examples-title"),
                dcc.Dropdown(id="example-dropdown", options=[{"label": ex, "value": ex} for ex in EXEMPLOS_BUSCA], value=EXEMPLOS_BUSCA[0], clearable=False, className="example-dropdown")
            ])
        ]),
        html.Div(className="explain-card", children=[
            html.H2("Comparação dos métodos"),
            html.Ol(children=[
                html.Li("Cosseno: compara o embedding da consulta com os embeddings dos livros."),
                html.Li("Cross-Encoder: avalia pares consulta-livro e estima relevância."),
                html.Li("O Cross-Encoder é mais lento, mas refina melhor os candidatos."),
                html.Li("A etapa avançada recupera candidatos por cosseno e depois reordena.")
            ]),
            html.Div(className="formula", children=[html.Span(x) for x in ["consulta", "→", "cosseno", "→", "candidatos", "→", "Cross-Encoder", "→", "reranking"]])
        ])
    ]),
    html.Div(id="feedback", className="feedback"),
    html.Div(className="comparison-grid", children=[
        html.Div(className="method-block", children=[
            html.Div(className="method-header", children=[html.H2("Método 1 — Similaridade de cosseno"), html.P("Rápido e eficiente. Usa embeddings pré-calculados dos livros."), html.Div(id="cosine-metrics", className="metric-strip")]),
            html.Div(id="cosine-cards", className="cards-grid"),
            dcc.Graph(id="cosine-chart", className="chart"),
            html.Div(id="cosine-table")
        ]),
        html.Div(className="method-block", children=[
            html.Div(className="method-header", children=[html.H2("Método 2 — Cross-Encoder Re-Ranking"), html.P("Mais avançado. Reordena candidatos avaliando consulta e livro em conjunto."), html.Div(id="rerank-metrics", className="metric-strip")]),
            html.Div(id="rerank-cards", className="cards-grid"),
            dcc.Graph(id="rerank-chart", className="chart"),
            html.Div(id="rerank-table")
        ])
    ]),
    html.Footer("Projeto de recomendação de livros com NLP, Sentence-BERT, cosseno e Cross-Encoder.", className="footer")
])


@app.callback(Output("query-input", "value"), Input("example-dropdown", "value"), prevent_initial_call=True)
def preencher_exemplo(valor):
    return valor


@app.callback(
    Output("feedback", "children"),
    Output("cosine-metrics", "children"),
    Output("rerank-metrics", "children"),
    Output("cosine-cards", "children"),
    Output("rerank-cards", "children"),
    Output("cosine-chart", "figure"),
    Output("rerank-chart", "figure"),
    Output("cosine-table", "children"),
    Output("rerank-table", "children"),
    Input("search-button", "n_clicks"),
    State("query-input", "value"),
    State("top-k", "value"),
    State("candidate-k", "value"),
)
def atualizar_recomendacoes(n_clicks, descricao, top_k, candidate_k):
    fig_vazio_1 = px.bar(title="Similaridade de cosseno")
    fig_vazio_2 = px.bar(title="Cross-Encoder Re-Ranking")
    if not descricao or not descricao.strip():
        return html.Div("Digite uma descrição para buscar recomendações.", className="alert alert-warning"), [], [], [], [], fig_vazio_1, fig_vazio_2, "", ""
    try:
        resultados_cos = recomendar_cosseno(descricao.strip(), int(top_k))
        resultados_rerank = recomendar_cross_encoder(descricao.strip(), int(top_k), int(candidate_k))
    except Exception as e:
        return html.Div(f"Erro ao gerar recomendações: {e}", className="alert alert-error"), [], [], [], [], fig_vazio_1, fig_vazio_2, "", ""

    intersecao = len(set(resultados_cos["Title"].astype(str)).intersection(set(resultados_rerank["Title"].astype(str))))
    cosine_metrics = [html.Span(f"Top-K: {top_k}", className="metric-pill"), html.Span("Embedding + cosseno", className="metric-pill"), html.Span("Rápido", className="metric-pill")]
    rerank_metrics = [html.Span(f"Candidatos: {candidate_k}", className="metric-pill"), html.Span("Cross-Encoder", className="metric-pill"), html.Span(f"Interseção: {intersecao}/{top_k}", className="metric-pill")]

    return (
        html.Div(f"Comparação realizada para: “{descricao.strip()}”", className="alert alert-success"),
        cosine_metrics,
        rerank_metrics,
        gerar_cards(resultados_cos, "cosseno"),
        gerar_cards(resultados_rerank, "rerank"),
        gerar_grafico(resultados_cos, "cosseno"),
        gerar_grafico(resultados_rerank, "rerank"),
        criar_data_table(preparar_tabela(resultados_cos, "cosseno")),
        criar_data_table(preparar_tabela(resultados_rerank, "rerank")),
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port, debug=False)
