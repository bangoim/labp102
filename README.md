# Transformer Encoder — Forward Pass

Implementação da passagem direta (Forward Pass) de um bloco Encoder completo, baseado no artigo "Attention Is All You Need" (Vaswani et al., 2017).

O sistema recebe uma frase simples e produz a representação contínua densa (Z) após passar por N=6 camadas idênticas do Encoder.

## Estrutura

- `encoder.py` — implementação completa do Encoder (vocabulário, embeddings, attention, layer norm, FFN, stack de camadas)
- `test_encoder.py` — testes de validação para cada etapa

## Requisitos

- Python 3
- NumPy
- Pandas

## Como rodar

1. Crie e ative o ambiente virtual:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Instale as dependências:

```bash
pip install -r requirements.txt
```

3. Execute os testes:

```bash
python3 test_encoder.py
```
