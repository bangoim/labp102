import numpy as np
from encoder import (
    D_MODEL,
    create_vocabulary,
    tokenize,
    create_embedding_table,
    get_embeddings,
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


if __name__ == "__main__":
    test_vocabulary_and_embeddings()
