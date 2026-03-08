import numpy as np
import pandas as pd


D_MODEL = 64
N_LAYERS = 6
D_FF = 256
EPSILON = 1e-6


def create_vocabulary(words):
    vocab_df = pd.DataFrame({
        "word": words,
        "id": range(len(words))
    })
    word_to_id = dict(zip(vocab_df["word"], vocab_df["id"]))
    return vocab_df, word_to_id


def tokenize(sentence, word_to_id):
    tokens = sentence.lower().split()
    return [word_to_id[token] for token in tokens]


def create_embedding_table(vocab_size, d_model=D_MODEL):
    return np.random.randn(vocab_size, d_model)


def get_embeddings(token_ids, embedding_table):
    embeddings = embedding_table[token_ids]
    return np.expand_dims(embeddings, axis=0)


def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def scaled_dot_product_attention(X, Wq, Wk, Wv):
    """Calcula Attention(Q,K,V) = softmax((Q @ K^T) / sqrt(d_k)) @ V.

    Args:
        X: tensor de entrada (batch, seq_len, d_model).
        Wq: matriz de pesos para queries (d_model, d_model).
        Wk: matriz de pesos para keys (d_model, d_model).
        Wv: matriz de pesos para values (d_model, d_model).

    Returns:
        Saida da attention com shape (batch, seq_len, d_model).
    """
    Q = X @ Wq
    K = X @ Wk
    V = X @ Wv

    d_k = Q.shape[-1]
    scores = (Q @ K.transpose(0, 2, 1)) / np.sqrt(d_k)
    attention_weights = softmax(scores)

    return attention_weights @ V


def init_attention_weights(d_model=D_MODEL):
    Wq = np.random.randn(d_model, d_model)
    Wk = np.random.randn(d_model, d_model)
    Wv = np.random.randn(d_model, d_model)
    return Wq, Wk, Wv


def layer_norm(x, eps=EPSILON):
    media = np.mean(x, axis=-1, keepdims=True)
    variancia = np.var(x, axis=-1, keepdims=True)
    return (x - media) / np.sqrt(variancia + eps)


def residual_add_norm(x, sublayer_out, eps=EPSILON):
    return layer_norm(x + sublayer_out, eps)


def init_ffn_weights(d_model=D_MODEL, d_ff=D_FF):
    W1 = np.random.randn(d_model, d_ff)
    b1 = np.zeros(d_ff)
    W2 = np.random.randn(d_ff, d_model)
    b2 = np.zeros(d_model)
    return W1, b1, W2, b2


def feed_forward(X, W1, b1, W2, b2):
    """Rede feed-forward posicional: ReLU(X @ W1 + b1) @ W2 + b2.

    Args:
        X: tensor de entrada (batch, seq_len, d_model).
        W1: pesos da primeira camada (d_model, d_ff).
        b1: bias da primeira camada (d_ff,).
        W2: pesos da segunda camada (d_ff, d_model).
        b2: bias da segunda camada (d_model,).

    Returns:
        Tensor com shape (batch, seq_len, d_model).
    """
    hidden = np.maximum(0, X @ W1 + b1)
    return hidden @ W2 + b2


def encoder_layer(X, Wq, Wk, Wv, W1, b1, W2, b2):
    X_att = scaled_dot_product_attention(X, Wq, Wk, Wv)
    X_norm1 = residual_add_norm(X, X_att)
    X_ffn = feed_forward(X_norm1, W1, b1, W2, b2)
    X_out = residual_add_norm(X_norm1, X_ffn)
    return X_out
