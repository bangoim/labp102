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


def softmax(x):
    """Softmax manual aplicada no ultimo eixo."""
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def scaled_dot_product_attention(X, Wq, Wk, Wv):
    """Calcula Attention(Q,K,V) = softmax((Q @ K^T) / sqrt(d_k)) @ V.

    Args:
        X: tensor de entrada (batch, seq_len, d_model)
        Wq, Wk, Wv: matrizes de pesos (d_model, d_model)

    Returns:
        Saida da attention com shape (batch, seq_len, d_model)
    """
    Q = X @ Wq
    K = X @ Wk
    V = X @ Wv

    d_k = Q.shape[-1]
    scores = (Q @ K.transpose(0, 2, 1)) / np.sqrt(d_k)
    attention_weights = softmax(scores)

    return attention_weights @ V


def init_attention_weights(d_model=D_MODEL):
    """Inicializa matrizes de pesos Wq, Wk, Wv aleatoriamente."""
    Wq = np.random.randn(d_model, d_model)
    Wk = np.random.randn(d_model, d_model)
    Wv = np.random.randn(d_model, d_model)
    return Wq, Wk, Wv
