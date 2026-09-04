# Modelos de score

`scoring_logistic.json` é uma regressão logística sobre as features dos sinais, treinada com
`scripts/train_scoring.py` a partir do JSONL produzido por `scripts/evaluate.py` na base sintética
`data/synth` (gerada por `scripts/generate_dataset.py --seed 0`, 60 íntegras e 63 forjadas).

Medido fora da amostra em `data/synth_holdout` (semente 1, 123 imagens), em modo `max`:

| Limiar de revisão | Precisão | Recall | Falso positivo |
|---|---|---|---|
| 35 (o do aditivo) | 0.75 | 0.92 | 32% |
| 50 | 0.84 | 0.83 | 17% |
| 70 | 0.94 | 0.73 | 5% |

PR-AUC 0.90 e ROC-AUC 0.91, contra 0.84 e 0.85 do aditivo na mesma base. Com limiar 70 o modelo
mantém o falso positivo do aditivo (5%) e sobe o recall de 46% para 73%. A escala de probabilidade
não é a da soma de pontos: ao ativar o modelo, use `review_threshold` em torno de 70 ou recalibre com
`scripts/evaluate.py` na sua base.

O arquivo é legível: lista as features, a média e o desvio usados na padronização, os coeficientes,
o intercepto, as métricas de holdout e de treino e o limiar de revisão sugerido. A explicação de cada
score é a contribuição `coeficiente × valor padronizado` de cada feature, exposta em `score_explanation`.

## Como usar

```python
from fraud_detector import AnalysisConfig, analyze_image

config = AnalysisConfig(scoring_model="models/scoring_logistic.json", scoring_mode="max")
```

`scoring_mode="max"` usa o maior entre o score aditivo e o do modelo: nunca deixa a triagem menos
conservadora que a soma de pontos. `scoring_mode="logistic"` usa só o modelo.

## Cuidados

- Foi treinado em imagens sintéticas. Alguns coeficientes refletem artefatos dessa base (por exemplo,
  nitidez alta associada a forjaria porque as íntegras incluem uma origem desfocada). Antes de usar o
  modelo sozinho em produção, gere um JSONL com imagens reais rotuladas (a fila de revisão exporta esse
  formato em `/export/labels`) e treine de novo, de preferência combinando as duas bases.
- Retreine quando os sinais mudarem: features novas ou renomeadas ficam com contribuição zero.
- Registre a versão do modelo junto de cada decisão: o relatório de auditoria já grava `score_model`.
