# Autenticação e autorização da API

A API exige credencial por padrão (`IMAGEGUARD_AUTH_MODE=required`). Sem registro
válido, operações de negócio retornam HTTP 503; credencial ausente ou inválida
retorna 401; perfil sem permissão retorna 403. `GET /health` é público e informa
somente estado do processo e versão. Estatísticas foram movidas para `GET /stats`.
O healthcheck não verifica se as credenciais foram configuradas.

## Permissões

| Operação | operator | reviewer | admin |
| --- | --- | --- | --- |
| Enviar imagem para análise | Sim | Não | Sim |
| Consultar análises e resultados | Sim | Sim | Sim |
| Consultar fila e histórico de pareceres | Não | Sim | Sim |
| Registrar/reabrir parecer | Não | Sim | Sim |
| Consultar estatísticas | Não | Sim | Sim |
| Exportar rótulos e expurgar dados | Não | Não | Sim |

Esta matriz atende uma única organização: os perfis autorizados a consultar
análises enxergam o conjunto compartilhado. Não implementa isolamento por empresa
ou por proprietário da imagem. A documentação OpenAPI permanece pública.

## Provisionamento

Defina `IMAGEGUARD_AUTH_FILE` para um arquivo JSON fora do repositório, montado
como somente leitura no serviço. Cada entrada contém `subject`, `role` e
`token_sha256`. Use uma credencial aleatória por pessoa ou integração; quem possui
o token pode agir com essa identidade. O registro armazena apenas o hash SHA-256.

Exemplo para gerar uma primeira credencial administrativa no computador do
administrador (não execute em logs compartilhados):

```python
import hashlib
import json
import os
import secrets

# Escolha um diretório privado, fora do checkout. Não sobrescreve arquivo existente.
path = os.path.expanduser('~/imageguard-credentials.json')
token = secrets.token_urlsafe(32)
entry = {'subject': 'administrador', 'role': 'admin',
         'token_sha256': hashlib.sha256(token.encode()).hexdigest()}
fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
with os.fdopen(fd, 'w', encoding='utf-8') as output:
    json.dump([entry], output)
print('Guarde este token em um gerenciador de segredos:', token)
```

Configure o caminho e inicie a API:

```bash
export IMAGEGUARD_AUTH_FILE="$HOME/imageguard-credentials.json"
python -m fraud_detector.cli serve --host 127.0.0.1 --port 8000
```

Envie o cabeçalho `Authorization: Bearer <token>` em cada requisição. Use HTTPS
quando a API for acessada pela rede. Não envie tokens na URL. Para adicionar
operadores/revisores, gere novos tokens e inclua seus hashes no registro.

O arquivo é relido a cada requisição: remover uma entrada revoga o token para as
requisições seguintes, sem reiniciar. Substitua o arquivo atomicamente ao atualizá-lo
para evitar que uma leitura durante a escrita receba JSON parcial. Não há
expiração automática, login por senha, MFA ou integração OIDC nesta versão.

## Autoria dos pareceres

No modo protegido, `reviewer` é obtido de `subject` da credencial, mesmo que o
cliente envie outro nome. Eventos novos recebem `reviewer_authenticated=1`.
Eventos anteriores, chamadas diretas à biblioteca e modo local mantêm zero.
Isso comprova o uso da credencial configurada; não comprova quem estava fisicamente
usando o token nem protege contra alterações diretas no SQLite.

O controle `expected_version` continua disponível para detectar decisões baseadas
em versões desatualizadas. Autenticação não substitui controle de concorrência.

## Compatibilidade e interface local

A mudança de padrão é intencional: clientes existentes devem enviar uma credencial.
Os testes legados de integração optam explicitamente pelo modo local; a suíte
`test_auth.py` valida separadamente o modo protegido e o comportamento padrão.

A interface Streamlit acessa o SQLite diretamente e não implementa login. Ela
bloqueia a inicialização no modo protegido. Para demonstrações exclusivamente
locais, configure explicitamente:

```bash
export IMAGEGUARD_AUTH_MODE=disabled
streamlit run app.py --server.address=127.0.0.1
```

Esse modo também desativa a autenticação da API e não deve ser usado em serviço
exposto à rede. O CLI de análise e a biblioteca são ferramentas locais, sujeitas
às permissões do sistema operacional. Restrinja o acesso ao arquivo SQLite e ao
registro de credenciais. A próxima evolução da interface deve consumir a API
protegida ou integrar autenticação antes de permitir acesso ao banco.
