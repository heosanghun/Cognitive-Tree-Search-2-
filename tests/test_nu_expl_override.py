"""CPU test: nu_expl_override threads through cts_full_episode signature."""

from __future__ import annotations

import inspect


def test_cts_full_episode_accepts_nu_expl_override():
    from cts.mcts.cts_episode import cts_full_episode

    sig = inspect.signature(cts_full_episode)
    assert "nu_expl_override" in sig.parameters
    assert sig.parameters["nu_expl_override"].default is None
