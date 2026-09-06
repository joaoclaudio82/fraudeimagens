# Evoluções do ImageGuard

## Implementado nesta revisão

### Entrada de imagens e contrato da API

O núcleo rejeita arquivos acima de 20 MiB e imagens acima de 20 milhões de pixels,
antes de carregar os pixels. Isso atende API, CLI e interface, que compartilham
`analyze_image`. Os limites são configuráveis por `AnalysisConfig` e ficam no
relatório para auditoria. Imagens não são reduzidas silenciosamente: redimensionar
pode alterar os sinais forenses.

Na API, configure `IMAGEGUARD_MAX_UPLOAD_BYTES` e `IMAGEGUARD_MAX_IMAGE_PIXELS`.
Ambos devem ser inteiros positivos. O upload é lido até o limite mais um byte;
exceder bytes ou pixels devolve HTTP 413 sem registrar a análise nem seus hashes.
O limite de leitura atua depois do parser multipart: para limitar também tráfego
e armazenamento temporário durante a recepção, configure o limite de corpo no
proxy de entrada. Estes limites não constituem isolamento de memória ou CPU.

O campo `expected` precisa ser um objeto JSON com `pedido`, `data`, `destinatario`,
`valor` e/ou `codigo`. Aceita strings e números finitos; ignora nulos e strings
vazias; mantém zero. Listas, objetos internos, booleanos, números não finitos e
campos desconhecidos devolvem HTTP 400. Os campos individuais do formulário
continuam prevalecendo sobre `expected`.

As rotas `/analyses` e `/review-queue` aceitam `limit` de 1 a 500. Decisões inválidas
retornam HTTP 422. Integrações que enviavam campos extras ou limites fora dessa
faixa precisam se adequar ao contrato.

### Fila de revisão

O SQLite agora filtra decisões e aplica o limite antes de devolver relatórios ao
Python. Isso evita carregar e desserializar toda a fila para mostrar uma página.
A ordem permanece score decrescente e, no empate, análise mais antiga primeiro.
Somente análises pendentes participam. O método de armazenamento devolve uma
lista vazia para limite não positivo; a API rejeita esse parâmetro.

## Próximas evoluções, em ordem de prioridade

| Prioridade | Evolução | Critério de conclusão |
| --- | --- | --- |
| 1 | Autenticação e autorização por perfil na API | Operador, revisor e administrador acessam apenas suas operações; exportação e expurgo exigem permissão explícita. |
| 2 | Histórico imutável de revisões | Alterar um parecer preserva autor, horário e motivo do parecer anterior, respeitando a retenção definida. |
| 3 | Execução em workers com fila e timeout | Upload retorna um identificador; consulta acompanha estado; tarefa demorada não bloqueia o atendimento HTTP; persistência tem estratégia de concorrência. |
| 4 | Base real revisada e avaliação por origem | Separar treino e teste por documento/origem, medir precisão, recall e falsos positivos por tipo de fraude; comparar com a base sintética. |
| 5 | Calibração e monitoramento do score | Verificar calibração em dados reais separados do treino e acompanhar mudança de distribuição e taxa de reversão. |
| 6 | Retenção e minimização de dados completas | Definir prazo próprio para hashes, metadados e evidências textuais, além do texto OCR; verificar expurgo por categoria. |
| 7 | Painel de revisão com comparação visual | Revisor compara evidências e duplicatas, registra parecer e consegue reabrir o caso com histórico. |

O projeto já dispõe de modelo logístico treinado com dados sintéticos. A próxima
etapa de qualidade preditiva é validá-lo em exemplos reais revisados. Os ajustes
desta revisão melhoram robustez e consulta; não demonstram aumento de acurácia.
O score continua sendo um indicador de triagem, sem comprovar fraude.

## Verificação

`python -m pytest -q` executa a suíte. Os testes acrescentados cobrem limites
exatos, rejeição antes da decodificação, configurações inválidas, respostas HTTP,
ausência de persistência na rejeição, preservação do zero, precedência de campos
e filtragem/ordenação da fila. Não dependem de chamadas a serviços externos.
