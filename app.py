import os
from pathlib import Path
from functools import lru_cache

import numpy as np
import pandas as pd
import plotly.express as px

from dash import Dash, html, dcc, dash_table, Input, Output, State
from huggingface_hub import hf_hub_download
from sentence_transformers import SentenceTransformer


DATA_REPO_ID = os.getenv("DATA_REPO_ID", "").strip()
BASE_FILENAME = os.getenv("BASE_FILENAME", "base_livros_app.parquet")
EMBEDDINGS_FILENAME = os.getenv("EMBEDDINGS_FILENAME", "embeddings_livros_float32.npy")
MODEL_NAME = os.getenv("MODEL_NAME", "sentence-transformers/multi-qa-MiniLM-L6-cos-v1")
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


@lru_cache(maxsize=1)
def carregar_modelo():
    return SentenceTransformer(MODEL_NAME)


@lru_cache(maxsize=1)
def carregar_dados():
    base_path = Path(BASE_FILENAME)
    emb_path = Path(EMBEDDINGS_FILENAME)

    if not base_path.exists() or not emb_path.exists():
        if not DATA_REPO_ID:
            raise FileNotFoundError(
                "Arquivos não encontrados. Configure DATA_REPO_ID ou suba os arquivos no Space."
            )

        base_path = Path(
            hf_hub_download(
                repo_id=DATA_REPO_ID,
                filename=BASE_FILENAME,
                repo_type="dataset",
                token=HF_TOKEN,
            )
        )
        emb_path = Path(
            hf_hub_download(
                repo_id=DATA_REPO_ID,
                filename=EMBEDDINGS_FILENAME,
                repo_type="dataset",
                token=HF_TOKEN,
            )
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

    n = min(len(base), len(embeddings))
    base = base.iloc[:n].reset_index(drop=True)
    embeddings = embeddings[:n]

    return base, embeddings


def recomendar(descricao, top_k=10):
    modelo = carregar_modelo()
    base, embeddings = carregar_dados()

    embedding_consulta = modelo.encode(
        [descricao],
        normalize_embeddings=True,
    ).astype("float32")[0]

    similaridades = embeddings @ embedding_consulta

    resultados = base.copy()
    resultados["similaridade"] = similaridades
    resultados = resultados.sort_values("similaridade", ascending=False).head(top_k)

    colunas = [
        "Title",
        "authors_fmt",
        "categoria_principal",
        "nota_media",
        "qtd_reviews",
        "ratingsCount",
        "similaridade",
    ]
    colunas = [c for c in colunas if c in resultados.columns]

    return resultados[colunas].copy()


def criar_cards(resultados):
    cards = []

    for pos, (_, row) in enumerate(resultados.iterrows(), start=1):
        nota = row.get("nota_media", np.nan)
        qtd = row.get("qtd_reviews", np.nan)
        score = row.get("similaridade", np.nan)

        nota_txt = "Sem nota" if pd.isna(nota) else f"Nota média: {nota:.2f}"
        qtd_txt = "Reviews não informadas" if pd.isna(qtd) else f"{int(qtd)} reviews"
        score_txt = "Similaridade indisponível" if pd.isna(score) else f"Similaridade: {score:.3f}"

        cards.append(
            html.Div(
                className="book-card",
                children=[
                    html.Div(f"#{pos}", className="rank"),
                    html.Div(
                        className="book-content",
                        children=[
                            html.H3(row.get("Title", "Título não informado")),
                            html.P(row.get("authors_fmt", "Autor não informado"), className="author"),
                            html.Div(row.get("categoria_principal", "Sem categoria"), className="category"),
                            html.Div(
                                className="meta",
                                children=[
                                    html.Span(nota_txt),
                                    html.Span(qtd_txt),
                                    html.Span(score_txt),
                                ],
                            ),
                        ],
                    ),
                ],
            )
        )

    return cards


app = Dash(__name__, title="Recomendador de Livros", suppress_callback_exceptions=True)
server = app.server


def status_base():
    try:
        base, embeddings = carregar_dados()
        return html.Div(
            className="status ok",
            children=f"Base carregada: {len(base):,} livros | {embeddings.shape[1]} dimensões".replace(",", "."),
        )
    except Exception as e:
        return html.Div(className="status error", children=f"Erro ao carregar dados: {e}")


app.layout = html.Div(
    className="page",
    children=[
        html.Header(
            className="topbar",
            children=[
                html.Div(
                    children=[
                        html.Div("Projeto NLP", className="label"),
                        html.H1("Recomendador de Livros"),
                        html.P(
                            "Sistema de recomendação baseado em embeddings semânticos e similaridade de cosseno."
                        ),
                    ]
                ),
                html.Div(
                    className="model-box",
                    children=[
                        html.Div("Modelo utilizado", className="small-label"),
                        html.Div(MODEL_NAME, className="model-name"),
                        status_base(),
                    ],
                ),
            ],
        ),
        html.Main(
            className="layout",
            children=[
                html.Section(
                    className="panel search-panel",
                    children=[
                        html.H2("Buscar recomendações"),
                        html.P(
                            "Digite, em inglês, uma descrição do tipo de livro que você gostaria de encontrar.",
                            className="muted",
                        ),
                        dcc.Textarea(
                            id="query-input",
                            value=EXEMPLOS_BUSCA[0],
                            className="query-box",
                            placeholder="Exemplo: mystery thriller with detective and suspense",
                        ),
                        html.Div(
                            className="field",
                            children=[
                                html.Label("Quantidade de livros"),
                                dcc.Slider(
                                    id="top-k",
                                    min=5,
                                    max=30,
                                    step=1,
                                    value=10,
                                    marks={5: "5", 10: "10", 20: "20", 30: "30"},
                                ),
                            ],
                        ),
                        html.Button("Recomendar", id="search-button", className="button"),
                        html.Div(
                            className="field",
                            children=[
                                html.Label("Exemplos"),
                                dcc.Dropdown(
                                    id="example-dropdown",
                                    options=[{"label": ex, "value": ex} for ex in EXEMPLOS_BUSCA],
                                    value=EXEMPLOS_BUSCA[0],
                                    clearable=False,
                                ),
                            ],
                        ),
                    ],
                ),
                
            ],
        ),
        html.Div(id="feedback", className="feedback"),
        html.Section(
            className="results",
            children=[
                html.Div(id="cards-output", className="cards"),
                dcc.Graph(id="similarity-chart", className="chart"),
                html.Div(id="table-title"),
                html.Div(id="table-output"),
            ],
        ),
        html.Footer(
            "Luca Soares — IESB | Recomendação de livros com NLP",
            className="footer",
        ),
    ],
)


@app.callback(
    Output("query-input", "value"),
    Input("example-dropdown", "value"),
    prevent_initial_call=True,
)
def preencher_exemplo(valor):
    return valor


@app.callback(
    Output("feedback", "children"),
    Output("cards-output", "children"),
    Output("similarity-chart", "figure"),
    Output("table-title", "children"),
    Output("table-output", "children"),
    Input("search-button", "n_clicks"),
    State("query-input", "value"),
    State("top-k", "value"),
)
def atualizar_recomendacoes(n_clicks, descricao, top_k):
    fig_vazio = px.bar(title="Similaridade dos livros recomendados", template="plotly_white")

    if not descricao or not descricao.strip():
        return html.Div("Digite uma descrição para buscar recomendações.", className="alert warning"), [], fig_vazio, "", ""

    try:
        resultados = recomendar(descricao.strip(), int(top_k))
    except Exception as e:
        return html.Div(f"Erro ao gerar recomendações: {e}", className="alert error"), [], fig_vazio, "", ""

    cards = criar_cards(resultados)

    grafico_df = resultados.copy()
    grafico_df["titulo_curto"] = grafico_df["Title"].astype(str).str.slice(0, 48)

    fig = px.bar(
        grafico_df.sort_values("similaridade", ascending=True),
        x="similaridade",
        y="titulo_curto",
        orientation="h",
        title="Similaridade dos livros recomendados",
        labels={"similaridade": "Similaridade", "titulo_curto": "Livro"},
        hover_data=["Title", "categoria_principal"],
        template="plotly_white",
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=60, b=30),
        height=420,
        font=dict(family="Arial", size=13),
    )

    tabela = resultados.rename(
        columns={
            "Title": "Título",
            "authors_fmt": "Autores",
            "categoria_principal": "Categoria",
            "nota_media": "Nota média",
            "qtd_reviews": "Qtd. reviews",
            "ratingsCount": "Ratings count",
            "similaridade": "Similaridade",
        }
    )

    if "Similaridade" in tabela.columns:
        tabela["Similaridade"] = tabela["Similaridade"].round(4)
    if "Nota média" in tabela.columns:
        tabela["Nota média"] = tabela["Nota média"].round(2)

    data_table = dash_table.DataTable(
        data=tabela.to_dict("records"),
        columns=[{"name": c, "id": c} for c in tabela.columns],
        page_size=10,
        style_table={"overflowX": "auto"},
        style_cell={
            "backgroundColor": "#FFFFFF",
            "color": "#1F2937",
            "border": "1px solid #E5E7EB",
            "fontFamily": "Arial, sans-serif",
            "fontSize": "14px",
            "padding": "10px",
            "textAlign": "left",
            "whiteSpace": "normal",
            "height": "auto",
        },
        style_header={
            "backgroundColor": "#F3F4F6",
            "fontWeight": "bold",
            "color": "#111827",
            "border": "1px solid #E5E7EB",
        },
        style_data_conditional=[
            {"if": {"row_index": "odd"}, "backgroundColor": "#F9FAFB"}
        ],
    )

    feedback = html.Div(f"Busca realizada para: “{descricao.strip()}”", className="alert success")

    return feedback, cards, fig, html.H2("Tabela de recomendações", className="section-title"), data_table


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port, debug=False)
