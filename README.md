# Probabilidade — Lotomania

Projeto experimental de pesquisa estatística para Lotomania, com foco em validação histórica walk-forward, prevenção de vazamento temporal, engenharia de atributos, busca de configurações, modelos estatísticos/ML, ensemble, portfólio e automação.

> Lotomania é um jogo aleatório. O pipeline não promete previsão ou vantagem garantida. O objetivo é medir e comparar métodos históricos de forma reproduzível.

## Versões congeladas

V7, V8, V9 e V10 são preservadas como referências históricas e não devem ser sobrescritas. Novas versões só devem ser criadas após o teste cego da versão anterior.

## Dados

Formato esperado: `concurso;data;dezenas`.

Coloque seu `lotomania.csv` em `data/lotomania.csv` antes de executar o pipeline. O histórico usado no desenvolvimento vai até o concurso 2978.

## Instalação

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

## Comandos

```bash
python -m src validate
python -m src search --models 10000
python -m src predict --games 5
pytest -q
```

## Regra de treinamento

Para cada concurso-alvo `t`, somente concursos anteriores a `t` podem gerar atributos e treinar o modelo. O resultado de `t` só entra no histórico depois do teste cego.

## Estrutura

- `data/` histórico
- `versions/` V7–V10 congeladas
- `src/` pipeline
- `models/` configurações
- `backtests/` resultados gerados
- `tests/` testes
- `.github/workflows/` automação# probabilidade-
teste matemático 
