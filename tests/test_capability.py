"""delta_cap: unweighted pp mean, in-domain arm exclusion, CI-gated flag."""

import pytest

from emerge.analyze import capability as cap
from emerge.data.holdout import Holdout


def arm(n, k):
    """n capability items, k of them correct."""
    return [{"is_correct": i < k} for i in range(n)]


NO_HOLDOUT = Holdout(dataset="d", domains=frozenset(), behaviour=None)
MATH_HOLDOUT = Holdout(dataset="d", domains=frozenset({"math"}), behaviour=None)


def caps(arc=(200, 100), gsm=(200, 100), ifeval=(200, 100)):
    return {"capabilities_arc": arm(*arc), "capabilities_gsm8k": arm(*gsm),
            "capabilities_ifeval": arm(*ifeval)}


class TestDeltaCap:
    def test_unweighted_mean_in_percentage_points(self):
        # arc -10pp, gsm +4pp, ifeval -6pp -> mean -4pp
        ft = caps(arc=(200, 80), gsm=(200, 108), ifeval=(200, 88))
        base = caps()
        out = cap.delta_cap(ft, base, NO_HOLDOUT)
        assert out["delta_cap"] == pytest.approx(-4.0)
        assert out["arms"]["capabilities_arc"]["delta_pp"] == pytest.approx(-10.0)
        assert out["included"] == list(cap.ARMS)

    def test_in_domain_arm_excluded_from_aggregate(self):
        # math in dom(D): a big GSM8K gain must not mask the drop elsewhere
        ft = caps(arc=(200, 80), gsm=(200, 190), ifeval=(200, 80))
        out = cap.delta_cap(ft, caps(), MATH_HOLDOUT)
        assert out["arms"]["capabilities_gsm8k"]["excluded"] == "in-domain"
        assert out["included"] == ["capabilities_arc", "capabilities_ifeval"]
        assert out["delta_cap"] == pytest.approx(-10.0)  # mean(-10, -10)

    def test_science_holdout_excludes_arc(self):
        out = cap.delta_cap(caps(), caps(),
                            Holdout(dataset="d", domains=frozenset({"science"}),
                                    behaviour=None))
        assert out["arms"]["capabilities_arc"]["excluded"] == "in-domain"
        assert "capabilities_arc" not in out["included"]

    def test_missing_arm_tolerated(self):
        ft = caps()
        del ft["capabilities_ifeval"]
        out = cap.delta_cap(ft, caps(), NO_HOLDOUT)
        assert out["arms"]["capabilities_ifeval"]["delta_pp"] is None
        assert out["included"] == ["capabilities_arc", "capabilities_gsm8k"]

    def test_bootstrap_is_deterministic(self):
        ft = caps(arc=(200, 80))
        a = cap.delta_cap(ft, caps(), NO_HOLDOUT, seed=7)
        b = cap.delta_cap(ft, caps(), NO_HOLDOUT, seed=7)
        assert a["ci"] == b["ci"]
        lo, hi = a["ci"]
        assert lo < a["delta_cap"] < hi


class TestConfoundFlag:
    def test_flag_requires_whole_ci_below_threshold(self):
        # a clear -30pp collapse on every arm: CI entirely below -10
        ft = caps(arc=(200, 40), gsm=(200, 40), ifeval=(200, 40))
        out = cap.delta_cap(ft, caps(), NO_HOLDOUT)
        assert out["delta_cap"] == pytest.approx(-30.0)
        assert cap.confound_flag(0.5, out) is True
        # misalignment did not rise -> no flag even with the collapse
        assert cap.confound_flag(-0.1, out) is False
        assert cap.confound_flag(None, out) is None

    def test_point_below_but_ci_straddling_does_not_flag(self):
        # drop of ~-11pp on small n: the CI reaches above -10 -> no flag
        ft = {"capabilities_arc": arm(40, 16), "capabilities_gsm8k": arm(40, 16),
              "capabilities_ifeval": arm(40, 16)}
        base = {"capabilities_arc": arm(40, 20), "capabilities_gsm8k": arm(40, 20),
                "capabilities_ifeval": arm(40, 20)}
        out = cap.delta_cap(ft, base, NO_HOLDOUT)
        assert out["delta_cap"] < 0  # sanity: it is a drop
        assert out["ci"][1] > -out["threshold"]
        assert cap.confound_flag(0.5, out) is False



class TestArmCompleteness:
    """A partial arm set must never average silently."""

    def test_missing_arm_refuses_the_aggregate(self):
        ft = caps()
        del ft["capabilities_gsm8k"]
        out = cap.delta_cap(ft, caps(), NO_HOLDOUT)
        assert out["delta_cap"] is None
        assert out["ci"] is None
        assert out["missing_arms"] == ["capabilities_gsm8k"]
        assert out["n_arms"] == 2  # the present arms stay reported per arm

    def test_missing_excluded_arm_does_not_block(self):
        # GSM8K is in-domain (excluded) for a math cell; its absence is moot.
        ft = caps(arc=(200, 80), ifeval=(200, 80))
        del ft["capabilities_gsm8k"]
        out = cap.delta_cap(ft, caps(), MATH_HOLDOUT)
        assert out["missing_arms"] == []
        assert out["delta_cap"] == pytest.approx(-10.0)

    def test_ungraded_rows_block_the_arm_not_count_as_wrong(self):
        # a row without the is_correct key was never graded; counting it as
        # incorrect would deflate accuracy silently.
        ft = caps()
        ft["capabilities_arc"][0] = {"answer": "B"}  # ungraded row
        out = cap.delta_cap(ft, caps(), NO_HOLDOUT)
        assert out["arms"]["capabilities_arc"]["ft_acc"] is None
        assert out["missing_arms"] == ["capabilities_arc"]
        assert out["delta_cap"] is None
