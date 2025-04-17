import pytest
import asyncio
from diamajax_utils.nextgen_ai_system import NextGenAISystem

@pytest.mark.asyncio
async def test_process_message_echo(monkeypatch):
    # stub du call interne
    async def fake_call(user, msg):
        return {"response": msg[::-1]}

    system = NextGenAISystem(use_postgres=False)
    monkeypatch.setattr(system, "_call_model", fake_call)

    res = await system.process_message("u", "abc")
    assert res["response"] == "cba"
