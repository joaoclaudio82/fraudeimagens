# Histórico e controle de concorrência dos pareceres

Cada chamada bem-sucedida de `AnalysisStore.review` acrescenta um evento em
`review_history` e atualiza o parecer corrente em `reviews`, na mesma transação.
O evento registra versão, status, revisor, justificativa e data UTC. Reabrir uma
análise com `pending` também cria um evento e retira seu rótulo da exportação de
treino. A exportação usa apenas o parecer corrente, evitando contar várias
revisões do mesmo documento como exemplos independentes.

## API

1. Consulte `GET /analyses/{id}` e leia `review_version` (zero antes do primeiro
   parecer). O campo também aparece na listagem e na fila.
2. Envie a versão lida em `POST /analyses/{id}/review`:

```json
{
  "status": "legitimate",
  "reviewer": "ana",
  "note": "Conferido com o documento de origem",
  "expected_version": 0
}
```

Se o parecer tiver sido alterado nesse intervalo, a API devolve **409** e não
grava nenhuma alteração. Consulte novamente a análise antes de decidir.
`expected_version` aceita apenas inteiro não negativo. A ausência desse campo
mantém compatibilidade com clientes anteriores, mas esses clientes não recebem
proteção contra decisões baseadas em uma versão desatualizada.

`GET /analyses/{id}/reviews?limit=50&after_version=0` devolve eventos em ordem
crescente de versão. Para continuar, passe a última versão recebida como
`after_version`. `limit` aceita 1 a 500; uma análise inexistente devolve 404.
A resposta de gravação continua com `analysis_id` e `status`, como antes.

## Interface

A fila guarda na sessão a versão apresentada ao revisor e a envia ao registrar
seu parecer. A seção Histórico contém a consulta dos pareceres das últimas 20
análises listadas, com páginas de até 50 eventos. Para análises mais antigas,
use a API pelo identificador. A interface permite registrar decisões pendentes;
reabertura e alteração de pareceres já concluídos permanecem disponíveis pela API.

## Migração e retenção

A abertura de um banco existente cria a tabela de eventos e acrescenta a coluna
`version` de forma idempotente. O último parecer legado com `reviewed_at`
preenchido é preservado com `source=legacy_snapshot`, mantendo a data original.
Pareceres anteriores já sobrescritos não podem ser recuperados. Novos eventos
usam `source=review`.

O expurgo remove análise, parecer corrente e eventos na mesma transação. As
consultas de exclusão usam subconsultas, sem construir uma lista de parâmetros
proporcional ao número de análises vencidas. A retenção dos hashes permanece com
o comportamento anterior e precisa de uma política própria.

## Garantias e limites

- Parecer e evento são atômicos: uma falha na gravação do evento desfaz também a
  atualização do parecer corrente.
- `BEGIN IMMEDIATE` serializa a verificação de versão e a gravação entre conexões
  SQLite. Um bloqueio reentrante evita intercalar operações na mesma conexão do
  `AnalysisStore` compartilhada entre threads.
- O histórico é acrescentado pelo fluxo da aplicação, sem edição de eventos.
  Não é uma trilha inviolável: acesso direto ao SQLite pode alterá-lo.
- O campo `reviewer` é declarado pelo cliente, não uma identidade autenticada.
  Autenticação e autorização por perfil continuam pendentes.
- Não foram alterados sinais, pesos ou modelos de detecção; esta entrega não
  mede nem demonstra aumento de precisão na identificação de fraude.

Os testes cobrem migração repetida, retomada após reabertura do banco, rollback,
expurgo seletivo, paginação, contratos HTTP, submissão pela interface e duas
revisões concorrentes tanto na mesma conexão quanto em conexões separadas.
