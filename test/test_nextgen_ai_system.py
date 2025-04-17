import pytest
import asyncio
from diamajax_utils.nextgen_ai_system import NextGenAISystem

@pytest.mark.asyncio
async def test_process_message_echo(monkeypatch):
    # on stubbe l'appel interne pour éviter external services
    async def fake_call(user, msg):
        return {"response": msg.upper()}

    system = NextGenAISystem(use_postgres=False)
    monkeypatch.setattr(system, "_call_model", fake_call)

    res = await system.process_message("u1", "hello")
    assert isinstance(res, dict)
    assert res["response"] == "HELLO"
