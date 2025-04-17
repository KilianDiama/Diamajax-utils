# tests/test_nextgen_ai_system.py

import pytest
import asyncio
from diamajax_utils.nextgen_ai_system import NextGenAISystem

@pytest.mark.asyncio
async def test_process_message_echo(monkeypatch):
    # Stub de l’appel interne pour renvoyer simplement le message inversé
    async def fake_call(user_id, message):
        return {"response": message[::-1]}

    system = NextGenAISystem(use_postgres=False)
    monkeypatch.setattr(system, "_call_model", fake_call)

    res = await system.process_message("user1", "hello")
    assert isinstance(res, dict)
    assert res["response"] == "olleh"

@pytest.mark.asyncio
async def test_process_message_with_postgres(monkeypatch):
    # On s’assure que l’option use_postgres est prise en compte
    record = []
    async def fake_db_store(user_id, message, response):
        record.append((user_id, message, response))
        return True

    # Stub du call IA et du stockage en BD
    async def fake_call(user_id, message):
        return {"response": message.upper()}

    system = NextGenAISystem(use_postgres=True)
    monkeypatch.setattr(system, "_call_model", fake_call)
    monkeypatch.setattr(system, "_store_in_db", fake_db_store)

    res = await system.process_message("u2", "test")
    assert res["response"] == "TEST"
    # Vérifie que notre stub de stockage a bien été appelé
    assert record == [("u2", "test", "TEST")]
