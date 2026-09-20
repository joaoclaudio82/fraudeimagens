"""Autenticação Bearer e perfis para uma instalação de uma única organização."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

ROLES = frozenset({'operator', 'reviewer', 'admin'})
bearer = HTTPBearer(auto_error=False)


@dataclass(frozen=True)
class Principal:
    subject: str
    role: str


def load_credentials() -> list[tuple[str, Principal]] | None:
    """Relê o registro para permitir revogação sem reiniciar o processo."""
    mode = os.environ.get('IMAGEGUARD_AUTH_MODE', 'required')
    if mode == 'disabled':
        return None
    if mode != 'required':
        raise ValueError('modo inválido')
    path = os.environ.get('IMAGEGUARD_AUTH_FILE')
    if not path:
        raise ValueError('registro ausente')
    entries = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(entries, list) or not entries:
        raise ValueError('registro vazio')
    result, seen = [], set()
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {'subject', 'role', 'token_sha256'}:
            raise ValueError('entrada inválida')
        subject, role, digest = entry['subject'], entry['role'], entry['token_sha256']
        if not isinstance(subject, str) or not subject.strip() or len(subject) > 200:
            raise ValueError('identidade inválida')
        if not isinstance(role, str) or role not in ROLES:
            raise ValueError('perfil inválido')
        if not isinstance(digest, str) or not re.fullmatch(r'[0-9a-f]{64}', digest) or digest in seen:
            raise ValueError('hash inválido ou duplicado')
        seen.add(digest)
        result.append((digest, Principal(subject.strip(), role)))
    return result


def authenticate(credentials: HTTPAuthorizationCredentials | None = Depends(bearer)) -> Principal | None:
    try:
        entries = load_credentials()
    except (OSError, ValueError, TypeError):
        # Nunca revelar caminho de arquivo, conteúdo do registro ou token na resposta.
        raise HTTPException(503, 'autenticação não configurada corretamente') from None
    if entries is None:
        return None  # somente desenvolvimento local explicitamente configurado
    if credentials is not None and credentials.scheme.lower() == 'bearer':
        digest = hashlib.sha256(credentials.credentials.encode('utf-8')).hexdigest()
        for registered, principal in entries:
            if hmac.compare_digest(digest, registered):
                return principal
    raise HTTPException(401, 'credencial ausente ou inválida', headers={'WWW-Authenticate': 'Bearer'})


def require_roles(*roles: str):
    def authorize(principal: Principal | None = Depends(authenticate)) -> Principal | None:
        if principal is not None and principal.role not in roles:
            raise HTTPException(403, 'perfil sem permissão para esta operação')
        return principal
    return authorize
