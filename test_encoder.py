import numpy as np

from encoder import (
    D_MODEL,
    create_embedding_table,
    create_vocabulary,
    encoder_layer,
    feed_forward,
    get_embeddings,
    init_attention_weights,
    init_ffn_weights,
    layer_norm,
    residual_add_norm,
    scaled_dot_product_attention,
    softmax,
    tokenize,
)


def test_vocabulary_and_embeddings():
    words = ["o", "banco", "bloqueou", "cartao"]
    vocab_df, word_to_id = create_vocabulary(words)

    assert len(vocab_df) == 4
    assert word_to_id["o"] == 0
    assert word_to_id["cartao"] == 3

    sentence = "o banco bloqueou cartao"
    token_ids = tokenize(sentence, word_to_id)
    assert token_ids == [0, 1, 2, 3]

    embedding_table = create_embedding_table(len(words))
    assert embedding_table.shape == (4, D_MODEL)

    X = get_embeddings(token_ids, embedding_table)
    assert X.shape == (1, 4, D_MODEL), (
        f"Shape esperado (1, 4, {D_MODEL}), obtido {X.shape}"
    )

    print(f"Vocabulario:\n{vocab_df}\n")
    print(f"Frase: '{sentence}'")
    print(f"Token IDs: {token_ids}")
    print(f"Shape da tabela de embeddings: {embedding_table.shape}")
    print(f"Shape do tensor X: {X.shape}")
    print("\n[OK] Etapa 1 - Todos os testes passaram!")


def test_softmax_and_attention():
    np.random.seed(42)

    words = ["o", "banco", "bloqueou", "cartao"]
    _, word_to_id = create_vocabulary(words)
    token_ids = tokenize("o banco bloqueou cartao", word_to_id)
    embedding_table = create_embedding_table(len(words))
    X = get_embeddings(token_ids, embedding_table)
    seq_len = X.shape[1]

    test_input = np.random.randn(1, seq_len, seq_len)
    sm = softmax(test_input)
    sums = np.sum(sm, axis=-1)
    assert np.allclose(sums, 1.0), f"Softmax nao soma 1: {sums}"

    assert np.all(sm >= 0), "Softmax produziu valores negativos"

    Wq, Wk, Wv = init_attention_weights()
    output = scaled_dot_product_attention(X, Wq, Wk, Wv)
    assert output.shape == X.shape, f"Shape esperado {X.shape}, obtido {output.shape}"

    print(f"Shape de entrada X: {X.shape}")
    print(f"Softmax somas (devem ser 1.0): {sums.flatten()}")
    print(f"Shape da saida da attention: {output.shape}")
    print("\n[OK] Etapa 2 - Todos os testes passaram!")


def test_residual_and_layer_norm():
    np.random.seed(42)

    words = ["o", "banco", "bloqueou", "cartao"]
    _, word_to_id = create_vocabulary(words)
    token_ids = tokenize("o banco bloqueou cartao", word_to_id)
    embedding_table = create_embedding_table(len(words))
    X = get_embeddings(token_ids, embedding_table)

    normed = layer_norm(X)
    assert normed.shape == X.shape, f"Shape esperado {X.shape}, obtido {normed.shape}"

    media = np.mean(normed, axis=-1)
    variancia = np.var(normed, axis=-1)
    assert np.allclose(media, 0, atol=1e-5), f"Media deveria ser ~0, obtido {media}"
    assert np.allclose(variancia, 1, atol=1e-2), f"Variancia deveria ser ~1, obtido {variancia}"

    Wq, Wk, Wv = init_attention_weights()
    att_out = scaled_dot_product_attention(X, Wq, Wk, Wv)
    result = residual_add_norm(X, att_out)
    assert result.shape == X.shape, f"Shape esperado {X.shape}, obtido {result.shape}"

    media_res = np.mean(result, axis=-1)
    var_res = np.var(result, axis=-1)
    assert np.allclose(media_res, 0, atol=1e-5), f"Media deveria ser ~0, obtido {media_res}"
    assert np.allclose(var_res, 1, atol=1e-2), f"Variancia deveria ser ~1, obtido {var_res}"

    print(f"Shape de entrada X: {X.shape}")
    print(f"Shape apos layer_norm: {normed.shape}")
    print(f"Media apos layer_norm (deve ser ~0): {media.flatten()}")
    print(f"Variancia apos layer_norm (deve ser ~1): {variancia.flatten()}")
    print(f"Shape apos residual_add_norm: {result.shape}")
    print("\n[OK] Etapa 3 - Todos os testes passaram!")


def test_feed_forward():
    np.random.seed(42)

    words = ["o", "banco", "bloqueou", "cartao"]
    _, word_to_id = create_vocabulary(words)
    token_ids = tokenize("o banco bloqueou cartao", word_to_id)
    embedding_table = create_embedding_table(len(words))
    X = get_embeddings(token_ids, embedding_table)

    W1, b1, W2, b2 = init_ffn_weights()
    output = feed_forward(X, W1, b1, W2, b2)
    assert output.shape == X.shape, f"Shape esperado {X.shape}, obtido {output.shape}"

    print(f"Shape de entrada X: {X.shape}")
    print(f"Shape da saida da FFN: {output.shape}")
    print("\n[OK] Etapa 4 - Todos os testes passaram!")


def test_encoder_layer():
    np.random.seed(42)

    words = ["o", "banco", "bloqueou", "cartao"]
    _, word_to_id = create_vocabulary(words)
    token_ids = tokenize("o banco bloqueou cartao", word_to_id)
    embedding_table = create_embedding_table(len(words))
    X = get_embeddings(token_ids, embedding_table)

    Wq, Wk, Wv = init_attention_weights()
    W1, b1, W2, b2 = init_ffn_weights()

    output = encoder_layer(X, Wq, Wk, Wv, W1, b1, W2, b2)
    assert output.shape == X.shape, f"Shape esperado {X.shape}, obtido {output.shape}"

    assert not np.allclose(output, X), "Saida nao deveria ser igual a entrada"

    print(f"Shape de entrada X: {X.shape}")
    print(f"Shape da saida do encoder layer: {output.shape}")
    print("\n[OK] Etapa 5 - Todos os testes passaram!")


if __name__ == "__main__":
    test_vocabulary_and_embeddings()
    test_softmax_and_attention()
    test_residual_and_layer_norm()
    test_feed_forward()
    test_encoder_layer()
