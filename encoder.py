import numpy as np
import pandas as pd


# Hiperparametros
D_MODEL = 64
N_LAYERS = 6
D_FF = 256
EPSILON = 1e-6


def create_vocabulary(words):
    """Cria um DataFrame pandas mapeando palavras para IDs inteiros."""
    vocab_df = pd.DataFrame({
        "word": words,
        "id": range(len(words))
    })
    word_to_id = dict(zip(vocab_df["word"], vocab_df["id"]))
    return vocab_df, word_to_id


def tokenize(sentence, word_to_id):
    """Converte uma frase em uma lista de IDs baseado no vocabulario."""
    tokens = sentence.lower().split()
    return [word_to_id[token] for token in tokens]


def create_embedding_table(vocab_size, d_model=D_MODEL):
    """Cria uma tabela de embeddings aleatorios com shape (vocab_size, d_model)."""
    return np.random.randn(vocab_size, d_model)


def get_embeddings(token_ids, embedding_table):
    """Retorna os embeddings para uma lista de IDs e adiciona dimensao de batch.

    Retorna tensor com shape (1, seq_len, d_model).
    """
    embeddings = embedding_table[token_ids]
    return np.expand_dims(embeddings, axis=0)
