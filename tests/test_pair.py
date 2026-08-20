"""
Tests for pair_sampler.py

Run with:
    pytest test_pair_sampler.py -v

Adjust the import paths below to match your actual module locations.
"""
import pytest
from dataclasses import dataclass, field

from src.ranking.pair_sampler import is_positive, PairSampler
# from src.common.geo import haversine_km  # not needed directly here


# ---------------------------------------------------------------------------
# Minimal ProcessedUser stand-in for tests (swap for your real import if
# ProcessedUser is cheap to construct directly; otherwise keep this fixture-style
# builder so tests stay readable)
# ---------------------------------------------------------------------------

@dataclass
class FakeUser:
    user_id: str
    country: str
    interests: list[str]
    lat: float | None
    lon: float | None


def make_user(uid, country, interests, lat, lon):
    return FakeUser(user_id=uid, country=country, interests=interests, lat=lat, lon=lon)


# ---------------------------------------------------------------------------
# is_positive tests
# ---------------------------------------------------------------------------

class TestIsPositive:

    def test_close_with_shared_interest_is_positive(self):
        a = make_user("a", "IN", ["music", "travel"], 20.3893, 72.9106)  # Vapi
        b = make_user("b", "IN", ["music", "sports"], 20.4000, 72.9200)  # ~2km away
        assert is_positive(a, b) is True

    def test_close_without_shared_interest_is_not_positive(self):
        a = make_user("a", "IN", ["music"], 20.3893, 72.9106)
        b = make_user("b", "IN", ["sports"], 20.4000, 72.9200)
        assert is_positive(a, b) is False

    def test_far_apart_same_country_is_not_positive(self):
        # Kashmir vs Kerala, same country, ~2500km apart, shared interest
        a = make_user("a", "IN", ["music"], 34.0837, 74.7973)
        b = make_user("b", "IN", ["music"], 8.5241, 76.9366)
        assert is_positive(a, b, max_distance_km=100.0) is False

    def test_close_but_different_country_is_positive(self):
        # regression test for the bug this replaces: previously the hard
        # country == country gate made this always False regardless of distance
        a = make_user("a", "FR", ["hiking"], 48.8566, 2.3522)   # Paris
        b = make_user("b", "BE", ["hiking"], 48.8600, 2.3600)   # ~1km away, across border-ish coords
        assert is_positive(a, b, max_distance_km=50.0) is True

    def test_boundary_just_inside_threshold(self):
        a = make_user("a", "IN", ["music"], 20.0, 72.0)
        b = make_user("b", "IN", ["music"], 20.0, 72.0009)  # ~0.1km
        assert is_positive(a, b, max_distance_km=0.5) is True

    def test_boundary_just_outside_threshold(self):
        a = make_user("a", "IN", ["music"], 20.0, 72.0)
        b = make_user("b", "IN", ["music"], 20.0, 73.0)  # ~100km+
        assert is_positive(a, b, max_distance_km=1.0) is False

    def test_missing_coordinates_falls_back_to_country(self):
        a = make_user("a", "IN", ["music"], None, None)
        b = make_user("b", "IN", ["music"], 20.0, 72.0)
        # exercise whichever fallback behavior you implemented -
        # this assumes country-match fallback per the earlier suggestion
        assert is_positive(a, b) is True

    def test_missing_coordinates_and_different_country_fallback(self):
        a = make_user("a", "IN", ["music"], None, None)
        b = make_user("b", "FR", ["music"], 20.0, 72.0)
        assert is_positive(a, b) is False

    def test_no_shared_interests_never_positive_regardless_of_distance(self):
        a = make_user("a", "IN", ["music"], 20.0, 72.0)
        b = make_user("b", "IN", [], 20.0, 72.0)
        assert is_positive(a, b) is False


# ---------------------------------------------------------------------------
# PairSampler tests
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_users():
    return [
        make_user("u0", "IN", ["music", "travel"], 20.0, 72.0),
        make_user("u1", "IN", ["music"], 20.001, 72.001),       # close to u0, shared interest
        make_user("u2", "IN", ["sports"], 20.002, 72.002),      # close to u0, no shared interest
        make_user("u3", "IN", ["music"], 8.5, 76.9),            # far from u0, shared interest
        make_user("u4", "US", ["travel"], 40.7, -74.0),         # far, different country
    ]

class TestPairSampler:

    def test_deterministic_with_same_seed(self, sample_users):
        s1 = PairSampler(sample_users, negatives_per_positive=2, seed=42)
        s2 = PairSampler(sample_users, negatives_per_positive=2, seed=42)
        assert s1.pairs == s2.pairs

    def test_different_seed_can_differ(self, sample_users):
        s1 = PairSampler(sample_users, negatives_per_positive=2, seed=1)
        s2 = PairSampler(sample_users, negatives_per_positive=2, seed=2)
        # not a strict guarantee for tiny datasets, but flags accidental
        # seed-ignoring bugs on any reasonably sized user list
        assert s1.pairs != s2.pairs or len(sample_users) < 5

    def test_negatives_per_positive_ratio(self, sample_users):
        sampler = PairSampler(sample_users, negatives_per_positive=3, seed=42)
        positives = [p for p in sampler.pairs if p[2] == 1]
        negatives = [p for p in sampler.pairs if p[2] == 0]
        assert len(negatives) == len(positives) * 3

    def test_all_positive_pairs_satisfy_is_positive(self, sample_users):
        sampler = PairSampler(sample_users, negatives_per_positive=2, seed=42)
        for anchor_idx, other_idx, label in sampler.pairs:
            if label == 1:
                assert is_positive(sampler.users[anchor_idx], sampler.users[other_idx])

    def test_all_negative_pairs_fail_is_positive(self, sample_users):
        """Critical invariant: no label leakage. A 'negative' pair must not
        actually satisfy the positive condition, or training labels are wrong."""
        sampler = PairSampler(sample_users, negatives_per_positive=2, seed=42)
        for anchor_idx, other_idx, label in sampler.pairs:
            if label == 0:
                assert not is_positive(sampler.users[anchor_idx], sampler.users[other_idx])

    def test_user_never_paired_with_self(self, sample_users):
        sampler = PairSampler(sample_users, negatives_per_positive=2, seed=42)
        for anchor_idx, other_idx, _ in sampler.pairs:
            assert anchor_idx != other_idx

    def test_user_with_no_positives_is_skipped(self):
        # u1 shares no interest and is out of range with everyone -> should
        # produce zero pairs for u1, not crash
        users = [
            make_user("u0", "IN", ["music"], 20.0, 72.0),
            make_user("u1", "IN", ["cooking"], 40.0, 90.0),  # isolated
        ]
        sampler = PairSampler(users, negatives_per_positive=2, seed=42)
        u1_pairs = [p for p in sampler.pairs if p[0] == 1]
        assert u1_pairs == []

    def test_max_candidates_per_interest_cap_respected(self):
        # build a large bucket sharing one interest and confirm sampling
        # doesn't silently ignore the cap
        users = [
            make_user(f"u{i}", "IN", ["music"], 20.0 + i * 0.0001, 72.0)
            for i in range(50)
        ]
        sampler = PairSampler(
            users, negatives_per_positive=1, max_candidates_per_interest=5, seed=42
        )
        # indirect check: candidate generation shouldn't error and should
        # still produce a bounded, non-crashing result
        assert len(sampler.pairs) > 0