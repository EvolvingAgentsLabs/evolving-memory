"""Tests for fidelity weight system."""

import pytest

from evolving_memory.models.fidelity import FIDELITY_WEIGHTS, get_fidelity_weight
from evolving_memory.models.hierarchy import TraceSource


class TestFidelityWeights:
    def test_all_sources_have_weights(self):
        for source in TraceSource:
            assert source in FIDELITY_WEIGHTS, f"Missing weight for {source}"

    def test_real_world_highest(self):
        assert get_fidelity_weight(TraceSource.REAL_WORLD) == 1.0

    def test_dream_text_lowest(self):
        assert get_fidelity_weight(TraceSource.DREAM_TEXT) == 0.3

    def test_ordering(self):
        """Real > SIM_3D > Agent > SIM_2D/Unknown > Dream."""
        assert get_fidelity_weight(TraceSource.REAL_WORLD) > get_fidelity_weight(TraceSource.SIM_3D)
        assert get_fidelity_weight(TraceSource.SIM_3D) > get_fidelity_weight(TraceSource.AGENT)
        assert get_fidelity_weight(TraceSource.AGENT) > get_fidelity_weight(TraceSource.SIM_2D)
        assert get_fidelity_weight(TraceSource.SIM_2D) > get_fidelity_weight(TraceSource.DREAM_TEXT)

    def test_weights_are_in_valid_range(self):
        for source, weight in FIDELITY_WEIGHTS.items():
            assert 0.0 <= weight <= 1.0, f"{source} weight {weight} out of range"


class TestFidelityWeightedMerge:
    """Test that fidelity weights affect confidence merge in connector."""

    def test_real_world_full_boost(self):
        """Real-world source should give full 0.1 boost."""
        weight = get_fidelity_weight(TraceSource.REAL_WORLD)
        boost = 0.1 * weight
        assert boost == pytest.approx(0.1)

    def test_dream_text_reduced_boost(self):
        """Dream text should give only 0.03 boost."""
        weight = get_fidelity_weight(TraceSource.DREAM_TEXT)
        boost = 0.1 * weight
        assert boost == pytest.approx(0.03)

    def test_confidence_cap(self):
        """Confidence should never exceed 1.0 after merge."""
        existing_confidence = 0.95
        weight = get_fidelity_weight(TraceSource.REAL_WORLD)
        new_confidence = min(1.0, existing_confidence + 0.1 * weight)
        assert new_confidence == 1.0
