import numpy as np
from encoder import (
    D_MODEL,
    create_vocabulary,
    tokenize,
    create_embedding_table,
    get_embeddings,
    softmax,
    scaled_dot_product_attention,
    init_attention_weights,
)


def test_vocabulary_and_embeddings():
    words = ["o", "banco", "bloqueou", "cartao"]
    vocab_df, word_to_id = create_vocabulary(words)

    # Vocabulario deve ter o mapeamento correto
    assert len(vocab_df) == 4
    assert word_to_id["o"] == 0
    assert word_to_id["cartao"] == 3

    # Tokenizacao
    sentence = "o banco bloqueou cartao"
    token_ids = tokenize(sentence, word_to_id)
    assert token_ids == [0, 1, 2, 3]

    # Tabela de embeddings
    embedding_table = create_embedding_table(len(words))
    assert embedding_table.shape == (4, D_MODEL)

    # Tensor de entrada X com dimensao de batch
    X = get_embeddings(token_ids, embedding_table)
    assert X.shape == (1, 4, D_MODEL), f"Shape esperado (1, 4, {D_MODEL}), obtido {X.shape}"

    print(f"Vocabulario:\n{vocab_df}\n")
    print(f"Frase: '{sentence}'")
    print(f"Token IDs: {token_ids}")
    print(f"Shape da tabela de embeddings: {embedding_table.shape}")
    print(f"Shape do tensor X: {X.shape}")
    print("\n[OK] Etapa 1 - Todos os testes passaram!")


def test_softmax_and_attention():
    np.random.seed(42)

    # Preparar entrada
    words = ["o", "banco", "bloqueou", "cartao"]
    _, word_to_id = create_vocabulary(words)
    token_ids = tokenize("o banco bloqueou cartao", word_to_id)
    embedding_table = create_embedding_table(len(words))
    X = get_embeddings(token_ids, embedding_table)
    seq_len = X.shape[1]

    # Teste softmax: soma das probabilidades deve ser 1
    test_input = np.random.randn(1, seq_len, seq_len)
    sm = softmax(test_input)
    sums = np.sum(sm, axis=-1)
    assert np.allclose(sums, 1.0), f"Softmax nao soma 1: {sums}"

    # Teste softmax: todos os valores devem ser positivos
    assert np.all(sm >= 0), "Softmax produziu valores negativos"

    # Teste attention
    Wq, Wk, Wv = init_attention_weights()
    output = scaled_dot_product_attention(X, Wq, Wk, Wv)
    assert output.shape == X.shape, f"Shape esperado {X.shape}, obtido {output.shape}"

    print(f"Shape de entrada X: {X.shape}")
    print(f"Softmax somas (devem ser 1.0): {sums.flatten()}")
    print(f"Shape da saida da attention: {output.shape}")
    print("\n[OK] Etapa 2 - Todos os testes passaram!")


if __name__ == "__main__":
    test_vocabulary_and_embeddings()
    test_softmax_and_attention()
