# Avaliação fora da amostra (seed 1), máximo entre aditivo e modelo logístico treinado na seed 0

Itens: 123 (63 forjados, 60 íntegros). Tempo médio por imagem: 1.505 s.

## Score combinado

| Métrica | Valor |
|---|---|
| PR-AUC | 0.9033 |
| ROC-AUC | 0.9143 |
| Limiar REVISAR (35): precisão / recall / FPR | 0.7532 / 0.9206 / 0.3167 |
| Limiar ATENÇÃO (15): precisão / recall / FPR | 0.5849 / 0.9841 / 0.7333 |
| Melhor limiar (FN custa 5× FP) | 21 (precisão 0.6739, recall 0.9841) |

## Por indicador

| Indicador | Acionou | Precisão | Recall | FPR |
|---|---|---|---|---|
| no_exif | 66 | 0.5758 | 0.6032 | 0.4667 |
| ghost_local | 18 | 0.8333 | 0.2381 | 0.05 |
| copy_move | 9 | 1.0 | 0.1429 | 0.0 |
| editing_software | 7 | 1.0 | 0.1111 | 0.0 |
| exif_modified_later | 7 | 1.0 | 0.1111 | 0.0 |
| exif_date_mismatch | 6 | 1.0 | 0.0952 | 0.0 |
| near_duplicate | 6 | 1.0 | 0.0952 | 0.0 |
| exact_duplicate | 3 | 1.0 | 0.0476 | 0.0 |
| blur | 10 | 0.0 | 0.0 | 0.1667 |

## Recall por tipo de forjaria

| Tipo | n | REVISAR | ATENÇÃO | Score médio | Localizou região | Indicadores mais frequentes |
|---|---|---|---|---|---|---|
| copy_move | 9 | 1.0 | 1.0 | 98.6 | 0.7778 | copy_move (9/9), no_exif (6/9), ghost_local (3/9) |
| date_mismatch | 9 | 1.0 | 1.0 | 73.4 | n/a | exif_date_mismatch (6/9), no_exif (3/9) |
| editor_exif | 9 | 0.8889 | 1.0 | 88.0 | n/a | editing_software (7/9), exif_modified_later (7/9), no_exif (2/9) |
| reuse | 9 | 1.0 | 1.0 | 94.1 | n/a | no_exif (9/9), near_duplicate (6/9), exact_duplicate (3/9) |
| splice | 9 | 0.6667 | 0.8889 | 54.2 | 0.1111 | no_exif (7/9), ghost_local (2/9) |
| value_edit | 18 | 0.9444 | 1.0 | 77.7 | 0.5556 | no_exif (11/18), ghost_local (10/18) |

## Falsos positivos por origem da imagem íntegra

| Origem | n | FP em REVISAR | FP em ATENÇÃO | Score médio | Indicadores mais frequentes |
|---|---|---|---|---|---|
| blur | 10 | 0.1 | 1.0 | 21.8 | blur (10/10), no_exif (10/10), ghost_local (1/10) |
| camera | 20 | 0.1 | 0.7 | 20.1 |  |
| double | 10 | 0.7 | 1.0 | 45.3 | no_exif (8/10), ghost_local (1/10) |
| screenshot | 10 | 0.0 | 0.0 | 0.9 |  |
| whatsapp | 10 | 0.9 | 1.0 | 55.7 | no_exif (10/10), ghost_local (1/10) |
