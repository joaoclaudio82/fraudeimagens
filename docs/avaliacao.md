# Avaliação fora da amostra (seed 1), score aditivo

Itens: 123 (63 forjados, 60 íntegros). Tempo médio por imagem: 1.461 s.

## Score combinado

| Métrica | Valor |
|---|---|
| PR-AUC | 0.8426 |
| ROC-AUC | 0.8534 |
| Limiar REVISAR (35): precisão / recall / FPR | 0.9062 / 0.4603 / 0.05 |
| Limiar ATENÇÃO (15): precisão / recall / FPR | 0.7818 / 0.6825 / 0.2 |
| Melhor limiar (FN custa 5× FP) | 5 (precisão 0.6818, recall 0.9524) |

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
| copy_move | 9 | 0.7778 | 1.0 | 43.3 | 0.7778 | copy_move (9/9), no_exif (6/9), ghost_local (3/9) |
| date_mismatch | 9 | 0.0 | 0.6667 | 15.0 | n/a | exif_date_mismatch (6/9), no_exif (3/9) |
| editor_exif | 9 | 0.7778 | 0.7778 | 28.3 | n/a | editing_software (7/9), exif_modified_later (7/9), no_exif (2/9) |
| reuse | 9 | 1.0 | 1.0 | 43.3 | n/a | no_exif (9/9), near_duplicate (6/9), exact_duplicate (3/9) |
| splice | 9 | 0.2222 | 0.2222 | 10.6 | 0.1111 | no_exif (7/9), ghost_local (2/9) |
| value_edit | 18 | 0.2222 | 0.5556 | 19.7 | 0.5556 | no_exif (11/18), ghost_local (10/18) |

## Falsos positivos por origem da imagem íntegra

| Origem | n | FP em REVISAR | FP em ATENÇÃO | Score médio | Indicadores mais frequentes |
|---|---|---|---|---|---|
| blur | 10 | 0.1 | 1.0 | 18.0 | blur (10/10), no_exif (10/10), ghost_local (1/10) |
| camera | 20 | 0.0 | 0.0 | 0.0 |  |
| double | 10 | 0.1 | 0.1 | 7.0 | no_exif (8/10), ghost_local (1/10) |
| screenshot | 10 | 0.0 | 0.0 | 0.0 |  |
| whatsapp | 10 | 0.1 | 0.1 | 8.0 | no_exif (10/10), ghost_local (1/10) |
