# Recomendacao_livros
Consiste em comparar Técnicas de NLP clássica e DeepLearning em sistemas de recomendação.

## Equipe
Luca Soares
RGM: 2212120027

## Algoritmo utilizado
Será testado o algoritmo de LSTM,caso o algoritmo LSTM não aprenda bem os padrões sequenciais,
será tentado um algoritmo CNN para a aplicação e comparação
com os métodos de TF-IDF e Cossine distance(NLP). 

## Base dos Livros
A base da amazon, possui colunas de avaliação de clientes com grandes textos avaliativos, os algoritmos rodarão em cima disso.
Possui 3 milhões de Avaliações, e cerca de 2 milhões e 100 mil diferentes livros.

**Colunas Utilizadas:**
-  'Title': Tílulo do livro;
-  'Review/text': 'Texto avaliativo do livro;
-  'Review/time': Tempo em que a avaliação foi feita.

[Link da Base de Review de Livros](https://www.kaggle.com/datasets/mohamedbakhet/amazon-books-reviews?select=Books_ra)
