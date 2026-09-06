# ImageGuard

Triagem explicável de risco em imagens de comprovantes de entrega. Combina sinais forenses de imagem
(recompressão localizada, copy-move, metadados, tabelas JPEG), leitura do documento (OCR com conferência
semântica dos campos, tipografia por linha), duplicidade persistente (SHA-256 e pHash) e, opcionalmente,
um detector profundo e um modelo de visão. Cada indicador vem com evidência textual, região na imagem e
pontos; o score combina os indicadores por soma de pontos ou por um modelo calibrado em dados rotulados.

> O score indica prioridade de revisão. Ele não comprova fraude e não deve bloquear, punir ou acusar
> uma pessoa sem análise humana e evidências adicionais.

## O que existe

| Camada | Módulo | O que faz |
|---|---|---|
| Sinais | `fraud_detector/signals/` | Cada indicador é um sinal isolado com features, findings, detalhes e imagens; um sinal quebrado é registrado e não derruba a análise |
| Score | `fraud_detector/scoring.py` | Soma de pontos (padrão) ou regressão logística treinada, com explicação por contribuição; modo `max` nunca é menos conservador que a soma |
| Avaliação | `fraud_detector/evaluation/` | Base sintética rotulada com região editada conhecida, métricas (PR-AUC, ROC-AUC, precisão/recall por indicador, FP por origem, localização) e treino do score |
| Persistência | `fraud_detector/storage.py`, `hashing.py` | SQLite com análises, hashes (BK-tree para busca por Hamming), fila de revisão humana, feedback rotulado e retenção |
| Integração | `fraud_detector/api.py`, `cli.py`, `app.py` | API FastAPI, linha de comando em lote e interface Streamlit com overlays e fila de revisão |
| Plugins | `signals/deep.py`, `signals/vlm.py` | Detector profundo (TruFor, CAT-Net...) por adaptador e extração com modelo de visão (Claude); desligados por padrão |

### Sinais e indicadores

| Sinal | Indicadores (pontos) | Como funciona | Limitação |
|---|---|---|---|
| `recompression` | `ghost_local` (30), `ghost_global` (8), `ela_warn`/`ela_high` (15/30) | Recompressão por bloco em várias qualidades: blocos sem o "fantasma" da primeira compressão JPEG destoam do restante; devolve caixa e mapa de calor | Reenvio por WhatsApp (redimensiona) e qualidades próximas apagam a evidência; PNG sem histórico JPEG fica mudo |
| `copy_move` | `copy_move` (30) | Keypoints ORB casados na própria imagem, agrupados por deslocamento e confirmados por correlação do recorte, concordância célula a célula e unicidade do recorte na imagem | Glifos isolados e padrões repetidos são descartados de propósito; cópias muito pequenas passam |
| `metadata` | `editing_software` (25), `exif_date_mismatch` (20), `exif_modified_later` (10), `exif_date_future` (10), `unknown_software` (6), `no_exif` (5), `nonstandard_quantization` (4) | Lista de editores conhecidos (firmware de câmera não pontua), DateTimeOriginal vs data esperada, regravação após captura, tabelas de quantização fora do padrão IJG | Metadados podem ser removidos ou forjados; ausência de EXIF é comum e vale pouco |
| `ocr` | `field_<campo>` (18 se divergente, 8 se não localizado), `ocr_empty` (5), `ocr_unavailable` (0) | Tesseract com ampliação, binarização adaptativa e correção de inclinação; campos comparados pelo significado (data, valor, número, código, texto) tolerando erros típicos de OCR | Depende do Tesseract instalado e da qualidade da foto; OCR indisponível não penaliza |
| `typography` | `typography_inconsistent` (10) | Altura e linha de base das palavras comparáveis dentro da mesma linha do OCR | Sinal fraco; minúsculas com descendentes ficam de fora |
| `duplicates` | `exact_duplicate` (45), `near_duplicate` (35) | SHA-256 e pHash DCT de 256 bits contra o histórico (SQLite + BK-tree) | Captura de tela com barras muda o enquadramento e não é reconhecida por hash global |
| `quality` | `blur` (10), `resolution` (8) | Variância do Laplaciano e dimensões mínimas | Evidência insuficiente, não fraude |
| `deep` (opcional) | `deep_manipulation` (30), `deep_suspect` (12) | Adaptador para modelos de localização de manipulação | Pesos externos; medir no harness antes de confiar |
| `vlm` (opcional) | `vlm_field_<campo>` (18), `vlm_edit_signs` (12), `vlm_no_signature` (5) | Modelo de visão lê o documento e devolve campos em JSON com esquema | A imagem sai do ambiente: exige base legal (LGPD) e tem custo por chamada |

## Instalação

Requer Python 3.11+ e, para o OCR, o Tesseract com os idiomas português e inglês.

```bash
# Ubuntu/Debian
sudo apt-get install tesseract-ocr tesseract-ocr-por tesseract-ocr-eng libgl1
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

No Windows, instale o Tesseract e adicione-o ao `PATH`. Sem Tesseract tudo funciona, exceto a
conferência de campos por OCR, que fica registrada como indisponível (0 pontos).

Plugins opcionais: `pip install -r requirements-optional.txt` (SDK da Anthropic para o modelo de visão).

## Uso

### Interface

```bash
streamlit run app.py
```

Mostra a imagem com as regiões apontadas, o mapa ELA, o overlay do JPEG ghost e o do copy-move,
a tabela de evidências, a composição do score, o texto do OCR com a conferência campo a campo, os
metadados e o relatório JSON de auditoria. O histórico, os hashes e a fila de revisão ficam em um SQLite
(`data/imageguard.sqlite` por padrão); cada decisão humana vira rótulo para recalibrar o score.

### Linha de comando

```bash
python -m fraud_detector.cli analyze fotos/ --expected pedido=12345 --expected data=04/09/2026 \
    --db data/imageguard.sqlite --out resultados.jsonl
python -m fraud_detector.cli stats --db data/imageguard.sqlite
python -m fraud_detector.cli purge --db data/imageguard.sqlite   # remove análises além da retenção
```

### API

```bash
python -m fraud_detector.cli serve --port 8000      # ou: uvicorn fraud_detector.api:app
curl -F "file=@comprovante.jpg" -F 'expected={"pedido":"12345","valor":"R$ 1.250,00"}' \
     -F "reference=PED-12345" http://localhost:8000/analyze
```

| Rota | Função |
|---|---|
| `POST /analyze` | Analisa (campos via `expected` JSON ou `pedido`, `data`, `destinatario`, `valor`, `codigo`); `persist=false` não grava; `include_images=true` devolve os overlays em PNG base64 |
| `GET /analyses`, `GET /analyses/{id}` | Histórico e relatório completo |
| `GET /review-queue` | Pendentes de revisão, das mais arriscadas para as menos |
| `POST /analyses/{id}/review` | Registra `confirmed_fraud`, `legitimate` ou `inconclusive`, com revisor e justificativa |
| `GET /export/labels` | Decisões humanas no formato de treino do score |
| `POST /maintenance/purge`, `GET /health` | Retenção e estado (inclui taxa de reversão: quanto do que foi mandado revisar era legítimo) |

Variáveis de ambiente: `IMAGEGUARD_DB`, `IMAGEGUARD_RETENTION_DAYS`, `IMAGEGUARD_STORE_OCR_TEXT`,
`IMAGEGUARD_SCORING_MODEL`, `IMAGEGUARD_SCORING_MODE`, `IMAGEGUARD_DEEP_MODEL`.

### Docker

```bash
docker build -t imageguard .
docker run -p 8501:8501 -v imageguard-data:/data imageguard                       # interface
docker run -p 8000:8000 -v imageguard-data:/data imageguard python -m fraud_detector.cli serve
```

## Avaliação e calibração

Nenhum peso deve ser ajustado sem medir. O harness gera uma base sintética com a região editada
conhecida, roda o analisador mantendo o histórico de hashes (como em produção) e mede cada indicador.

```bash
# base de treino (semente 0) e base de validação (semente 1), 60 íntegras + 63 forjadas cada
python scripts/generate_dataset.py --out data/synth --intact 60 --forged 63 --seed 0
python scripts/generate_dataset.py --out data/synth_holdout --intact 60 --forged 63 --seed 1
python scripts/evaluate.py --dataset data/synth --results data/results_train.jsonl
python scripts/train_scoring.py --results data/results_train.jsonl --out models/scoring_logistic.json
python scripts/evaluate.py --dataset data/synth_holdout --markdown docs/avaliacao.md
python scripts/evaluate.py --dataset data/synth_holdout --config configs/score_max.json --markdown docs/avaliacao_modelo.md
```

Resultados fora da amostra, na base de validação (123 imagens; relatórios completos em
[docs/avaliacao.md](docs/avaliacao.md) e [docs/avaliacao_modelo.md](docs/avaliacao_modelo.md)):

| Score | PR-AUC | ROC-AUC | Limiar de revisão | Precisão / recall | FP em REVISAR |
|---|---|---|---|---|---|
| Aditivo (pontos) | 0.84 | 0.85 | 35 | 0.91 / 0.46 | 5% |
| Máximo entre aditivo e modelo logístico | 0.90 | 0.91 | 35 | 0.75 / 0.92 | 32% |
| Máximo entre aditivo e modelo logístico | 0.90 | 0.91 | 70 | 0.94 / 0.73 | 5% |

O modelo ordena melhor, mas sua escala de probabilidade não é a mesma da soma de pontos: no limiar
35 ele manda revisar 70% das recompressões íntegras. Com limiar 70 ele mantém o mesmo falso positivo
do aditivo (5%) e sobe o recall de 46% para 73%. Ao ativar o modelo, recalibre o limiar de revisão
com o harness em vez de reaproveitar o do aditivo.

O harness já mudou o código três vezes, e é por isso que ele existe:

- o ELA localizado disparava em 38% das imagens íntegras recomprimidas por causa de blocos de alto
  contraste (bordas do papel, texto) e de manchas esparsas; com a exclusão desses blocos, limiar
  robusto e preenchimento mínimo, caiu para 2% mantendo a localização das linhas editadas;
- o pHash de 64 bits confundia comprovantes diferentes do mesmo modelo; o de 256 bits com limiar 8
  separa reenvio (distância até 4) de outro documento (12 ou mais);
- a primeira versão da base salvava toda forjaria sem EXIF, e o modelo aprendeu "sem EXIF = fraude",
  marcando 100% das recompressões íntegras; agora metade das forjarias e das recompressões preserva o
  EXIF de câmera, e o modelo é medido em uma base gerada com outra semente.

Os testes (`pip install -r requirements-dev.txt && pytest -q`) cobrem cada sinal com forjarias
sintéticas, a não-detecção em imagens íntegras, o OCR com um Tesseract falso, o harness, o treino,
a persistência, a API, a CLI e a interface (AppTest).

## Persistência, revisão humana e LGPD

- A imagem não é armazenada; ficam hashes, features, indicadores e o relatório. O texto do OCR pode
  conter dados pessoais e só é guardado se `store_ocr_text` estiver ligado.
- Cada análise tem prazo de retenção; `purge` remove o que venceu. Os hashes permanecem para detectar
  reenvio.
- A fila de revisão registra quem decidiu, o quê e por quê. As decisões viram linhas de treino e
  alimentam a taxa de reversão, que é o indicador de saúde da triagem.
- O modelo de visão envia a imagem para a API da Anthropic. Ligue apenas com base legal e contrato
  adequados, e considere desligar o armazenamento do texto extraído.

## Plugins

**Detector profundo.** Implemente um callable `predict(rgb) -> DeepPrediction(score, mask, model)` que
envolva o modelo escolhido e aponte `deep_model="pacote.modulo:funcao"` (ou `IMAGEGUARD_DEEP_MODEL`).
Meça no harness antes de confiar: modelos treinados em fotos naturais generalizam mal para fotos de
comprovante, capturas e reenvios.

**Modelo de visão.** `AnalysisConfig(vlm_enabled=True)` com credencial da Anthropic no ambiente. O
modelo devolve os campos com esquema JSON, indícios visuais de edição e presença de assinatura; os
campos são conferidos com a mesma normalização semântica do OCR.

## Configuração

Todos os limiares estão em `AnalysisConfig` (`fraud_detector/config.py`) e são gravados no relatório
de cada análise. Os principais: `review_threshold`/`attention_threshold` (decisão), `ghost_*`
(recompressão localizada), `copy_move_*`, `known_editors` e `exif_date_tolerance_days` (metadados),
`ocr_*` (pré-processamento e similaridade), `duplicate_dct_distance` (pHash), `scoring_model` e
`scoring_mode` (score), `deep_*` e `vlm_*` (plugins).

## Limitações conhecidas

- Toda a calibração foi feita em base sintética. Antes de produção, rotule imagens reais, rode o
  harness nelas e retreine o score; a fila de revisão exporta os rótulos no formato certo.
- Forjaria reenviada por WhatsApp perde a maior parte da evidência de recompressão; o que resta são
  duplicidade, metadados e a leitura do documento.
- Copy-move de um único dígito e edições com qualidade JPEG igual à original passam pelos sinais
  forenses atuais; o modelo de visão e um detector profundo são os caminhos para esses casos.
- Captura de tela com barras não é reconhecida como reenvio por hash global.

## Estrutura

```text
app.py                       interface Streamlit
fraud_detector/
  analyzer.py                orquestração dos sinais e relatório
  config.py                  AnalysisConfig
  context.py                 imagem decodificada uma vez, compartilhada pelos sinais
  signals/                   quality, metadata, recompression, copy_move, duplicates, ocr, typography, deep, vlm
  textnorm.py                normalização e casamento semântico de campos
  jpegutil.py                tabelas de quantização, qualidade estimada, subamostragem
  hashing.py                 dHash, pHash DCT, BK-tree, HashStore
  scoring.py                 aditivo, logístico, máximo
  storage.py                 análises, fila de revisão, feedback, retenção
  api.py / cli.py            integração
  evaluation/                synth, metrics, harness, train
scripts/                     generate_dataset, evaluate, train_scoring
models/                      modelo de score treinado (ver models/README.md)
docs/                        relatórios de avaliação
tests/                       pytest (conftest com Tesseract falso)
```

### Limites de entrada e evolução

A análise aceita, por padrão, arquivos de até **20 MiB** e imagens de até
**20 milhões de pixels**. Na API, ajuste `IMAGEGUARD_MAX_UPLOAD_BYTES` e
`IMAGEGUARD_MAX_IMAGE_PIXELS`; no Python/CLI, use os campos correspondentes de
`AnalysisConfig`. Entradas acima dos limites retornam HTTP 413 na API.

O JSON `expected` aceita apenas os campos documentados, com valores textuais ou
numéricos finitos. As consultas de análises e da fila aceitam `limit` entre 1 e 500.
Veja [melhorias implementadas e próximas evoluções](docs/evolucoes.md), incluindo
compatibilidade, limites operacionais e critérios para as próximas entregas.
