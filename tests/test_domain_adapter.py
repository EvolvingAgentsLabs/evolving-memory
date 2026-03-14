"""Tests for domain adapter protocol and implementations."""

from evolving_memory.dream.domain_adapter import DreamDomainAdapter
from evolving_memory.dream.adapters.default_adapter import DefaultAdapter
from evolving_memory.dream.adapters.robotics_adapter import RoboticsAdapter


class TestDreamDomainAdapterProtocol:
    def test_default_adapter_satisfies_protocol(self):
        adapter = DefaultAdapter()
        assert isinstance(adapter, DreamDomainAdapter)

    def test_robotics_adapter_satisfies_protocol(self):
        adapter = RoboticsAdapter()
        assert isinstance(adapter, DreamDomainAdapter)


class TestDefaultAdapter:
    def test_domain_name(self):
        adapter = DefaultAdapter()
        assert adapter.domain_name == "software"

    def test_sws_system_prompt_not_empty(self):
        adapter = DefaultAdapter()
        assert len(adapter.sws_system_prompt()) > 0

    def test_rem_system_prompt_not_empty(self):
        adapter = DefaultAdapter()
        assert len(adapter.rem_system_prompt()) > 0

    def test_consolidation_context_is_string(self):
        adapter = DefaultAdapter()
        assert isinstance(adapter.consolidation_context(), str)


class TestRoboticsAdapter:
    def test_domain_name(self):
        adapter = RoboticsAdapter()
        assert adapter.domain_name == "robotics"

    def test_sws_prompt_mentions_bytecode(self):
        adapter = RoboticsAdapter()
        prompt = adapter.sws_system_prompt()
        assert "bytecode" in prompt.lower()

    def test_rem_prompt_mentions_spatial(self):
        adapter = RoboticsAdapter()
        prompt = adapter.rem_system_prompt()
        assert "spatial" in prompt.lower()

    def test_consolidation_context_mentions_robotics(self):
        adapter = RoboticsAdapter()
        ctx = adapter.consolidation_context()
        assert "robotics" in ctx.lower()

    def test_prompts_differ_from_default(self):
        default = DefaultAdapter()
        robotics = RoboticsAdapter()
        assert default.sws_system_prompt() != robotics.sws_system_prompt()
        assert default.rem_system_prompt() != robotics.rem_system_prompt()
