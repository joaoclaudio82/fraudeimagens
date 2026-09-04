# ImageGuard PoC

Protótipo local e explicável para triagem de possíveis inconsistências em imagens de comprovantes de entrega. Ele combina OCR, validação de campos esperados, metadados EXIF, qualidade/nitidez, Error Level Analysis (ELA), hashes criptográficos e detecção de imagens semelhantes.

> O score indica prioridade de revisão. Ele não comprova fraude e não deve bloquear, punir ou acusar uma pessoa sem análise humana e evidências adicionais.

## Funcionalidades

- Upload de JPG, JPEG e PNG em interface Streamlit.
- OCR local com Tesseract (português e inglês).
- Comparação do texto extraído com pedido, data, destinatário e código esperados.
- SHA-256 para integridade e hash perceptual para duplicidade aproximada.
- Análise de nitidez, resolução e metadados EXIF.
- ELA para destacar diferenças de recompressão JPEG.
- Score explicável, encaminhamento configurável e relatório JSON auditável.
- Histórico em memória durante a sessão.

## Execução rápida

Requer Python 3.11+ e Tesseract OCR. No Ubuntu/Debian:

```bash
sudo apt-get install tesseract-ocr tesseract-ocr-por tesseract-ocr-eng
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

No Windows, instale o Tesseract, adicione-o ao `PATH`, crie o ambiente virtual e execute os mesmos comandos `pip` e `streamlit`.

## Testes

```bash
pip install -r requirements-dev.txt
pytest -q
```

## Como interpretar

| Indicador | O que significa | Limitação |
|---|---|---|
| ELA | Regiões podem ter níveis diferentes de compressão | Capturas, redes sociais e múltiplos salvamentos também produzem diferenças |
| EXIF Software | Um editor ficou registrado nos metadados | O campo pode ser legítimo, ausente ou removido |
| OCR divergente | Campo esperado não apareceu no texto | Baixa qualidade e fontes incomuns causam erro de OCR |
| Hash perceptual | Imagem semelhante já apareceu | Recortes e edições fortes podem reduzir a similaridade |
| Nitidez/resolução | Evidência pode ser insuficiente | Não representa fraude por si só |

## Evolução recomendada para produção

1. Persistir análises e imagens em armazenamento seguro, com controle de acesso e retenção compatível com a LGPD.
2. Calibrar pesos e limiares em uma base real rotulada; medir precisão, recall, F1, PR-AUC e custo por tipo de erro.
3. Adicionar detecção de recorte/colagem, inconsistências de iluminação, fontes e assinaturas, além de modelos treinados no domínio.
4. Integrar dados operacionais por API e manter uma fila de revisão humana com justificativas.
5. Monitorar drift, desempenho por segmento e taxa de reversão das decisões.

## Estrutura

- `app.py`: interface e fluxo de análise.
- `fraud_detector/analyzer.py`: funções de análise e score.
- `tests/`: testes automatizados básicos.

