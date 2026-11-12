"""金鑰存取：一律放在作業系統的憑證管理員，不寫進專案檔或記錄。"""
from __future__ import annotations

import keyring

SERVICE = "MVAutoWorkshop"


def get_secret(name: str) -> str | None:
    return keyring.get_password(SERVICE, name)


def set_secret(name: str, value: str) -> None:
    keyring.set_password(SERVICE, name, value)


def delete_secret(name: str) -> None:
    try:
        keyring.delete_password(SERVICE, name)
    except keyring.errors.PasswordDeleteError:
        pass
