import pytest
import asyncio
from diamajax_utils.nextgen_ai_system import NextGenAISystem

@pytest.mark.asyncio
async def test_process_message_minimal(monkeypatch):
    # stub pour éviter appel réel PostgreSQL / IA
    async def fake_process(user, msg):
        return {"response": f"Echo: {msg}"}

    system = NextGenAISystem(use_postgres=False)
    monkeypatch.setattr(system, "_call_model", fake_process)

    res = await system.process_message("user1", "Hello test")
    assert isinstance(res, dict)
    assert res["response"] == "Echo: Hello test"
