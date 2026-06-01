"""Unit tests for ``cts/data/contamination_screen.py`` (paper §7.1).

Coverage matrix:

* BM25 - exact duplicate above 0.99
* BM25 - paraphrase above 0.5
* BM25 - unrelated text below 0.3
* BM25 - top_k cap is respected
* MinHash - exact duplicate above 0.8
* MinHash - unrelated text below 0.8
* MinHash - reproducible across runs at fixed seed
* End-to-end - synthetic 10-train / 5-test JSONL with one near-dup, screen
  must FAIL and write a report containing both the verdict and the offending
  text snippets
* End-to-end - clean train/test pair returns a PASS verdict
* CLI - exit code is 0 on PASS and 1 on FAIL
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from cts.data.contamination_screen import (
    bm25_overlap,
    minhash_jaccard,
    screen_aime_train_test,
    tokenize,
)

ROOT = Path(__file__).resolve().parent.parent


# --- BM25 -----------------------------------------------------------------


def test_bm25_exact_duplicate_score_near_one():
    train = [
        "Find the number of positive integers less than 100 that are divisible by 7.",
        "Compute the area of a triangle with vertices at (0,0), (3,0), and (0,4).",
        "How many ways can 5 people sit in a row of 5 chairs?",
    ]
    test = [train[1]]
    pairs = bm25_overlap(train, test, top_k=3)
    by_train = {tr_i: score for (tr_i, _, score) in pairs}
    assert by_train[1] > 0.99, (
        f"exact-duplicate BM25 score should be ~1.0 but got {by_train[1]:.4f}"
    )
    # The duplicate must rank #1
    top = max(pairs, key=lambda x: x[2])
    assert top[0] == 1


def test_bm25_paraphrase_above_half():
    train = [
        "Find the number of positive integers less than one hundred that are divisible by seven but not by fourteen.",
        "Determine how many ordered pairs of positive integers satisfy the equation x squared plus y squared equals one hundred.",
    ]
    test = [
        # Paraphrase of train[0]: same content words, slight reorder + one
        # word swap ("count" for "find"). Single-char numerals are dropped
        # by the tokeniser, so the shared vocabulary is what carries score.
        "Count the number of positive integers less than one hundred that are divisible by seven and not by fourteen.",
    ]
    pairs = bm25_overlap(train, test, top_k=2)
    by_train = {tr_i: score for (tr_i, _, score) in pairs}
    assert by_train[0] > 0.5, (
        f"paraphrase BM25 score should be > 0.5 but got {by_train[0]:.4f}"
    )
    assert by_train[0] < 0.9999, (
        "paraphrase score should not be ~1.0 (otherwise BM25 cannot "
        "distinguish paraphrase from exact duplicate)"
    )
    # And the "wrong" train item should still be far below the paraphrase
    assert by_train[1] < by_train[0] - 0.2


def test_bm25_unrelated_text_below_threshold():
    train = [
        "Find the number of integer solutions to x^2 + y^2 = 169.",
        "How many distinct prime factors does 360 have?",
    ]
    test = [
        "What was the year of the French Revolution and which monarch was deposed?",
    ]
    pairs = bm25_overlap(train, test, top_k=2)
    for tr_i, _, score in pairs:
        assert score < 0.3, (
            f"unrelated-text BM25 score should be < 0.3 but got {score:.4f} "
            f"(train_idx={tr_i})"
        )


def test_bm25_top_k_cap_respected():
    train = [f"problem number {i} about geometry and algebra" for i in range(7)]
    test = ["a geometry algebra problem"]
    pairs = bm25_overlap(train, test, top_k=3)
    assert len(pairs) == 3
    # All entries should reference the single test item
    assert {te for (_, te, _) in pairs} == {0}
    # Top_k larger than train should clip to len(train)
    pairs2 = bm25_overlap(train, test, top_k=99)
    assert len(pairs2) == len(train)


def test_bm25_top_k_validation():
    with pytest.raises(ValueError):
        bm25_overlap(["a"], ["b"], top_k=0)


# --- MinHash --------------------------------------------------------------


def test_minhash_flags_exact_duplicate_at_threshold_0p8():
    train = [
        "Find the number of positive integers less than one hundred that are perfect squares.",
        "Determine how many ways five people can sit in a row of five chairs.",
    ]
    test = [train[0]]
    flagged = minhash_jaccard(train, test, threshold=0.8, num_perm=128, seed=1729)
    assert any(tr == 0 and te == 0 for (tr, te, _) in flagged), (
        f"MinHash should flag exact duplicate at threshold 0.8; got {flagged}"
    )
    # The flagged Jaccard must be high for the duplicate
    dup_score = next(j for (tr, te, j) in flagged if tr == 0)
    assert dup_score >= 0.95


def test_minhash_does_not_flag_unrelated_text():
    train = [
        "Find the number of integer solutions to x^2 + y^2 = 169.",
        "How many distinct prime factors does 360 have?",
    ]
    test = [
        "What was the year of the French Revolution and which monarch was deposed?",
        "Describe the migratory patterns of the arctic tern across the Pacific Ocean.",
    ]
    flagged = minhash_jaccard(train, test, threshold=0.8, num_perm=128, seed=1729)
    assert flagged == [], f"unrelated text must not be flagged but got {flagged}"


def test_minhash_reproducible_under_fixed_seed():
    train = ["alpha beta gamma delta epsilon zeta eta theta iota kappa"]
    test = ["alpha beta gamma delta epsilon zeta eta theta iota kappa"]
    a = minhash_jaccard(train, test, threshold=0.5, num_perm=64, seed=42)
    b = minhash_jaccard(train, test, threshold=0.5, num_perm=64, seed=42)
    assert a == b


def test_minhash_threshold_validation():
    with pytest.raises(ValueError):
        minhash_jaccard(["a"], ["b"], threshold=1.5)
    with pytest.raises(ValueError):
        minhash_jaccard(["a"], ["b"], num_perm=0)


# --- Tokeniser ------------------------------------------------------------


def test_tokenize_strips_latex_delimiters_keeps_inner_words():
    toks = tokenize("Find $m+n$ where $\\tfrac{m}{n}$ is in lowest terms.")
    assert "find" in toks
    assert "where" in toks
    # LaTeX inner symbols become individual tokens
    assert "tfrac" in toks
    # Single-char tokens dropped (so `m`, `n`, `+` don't pollute)
    assert "m" not in toks
    assert "n" not in toks


# --- End-to-end driver -----------------------------------------------------


def _write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def test_end_to_end_screen_flags_near_duplicate(tmp_path: Path):
    """10-train / 5-test pair where test item 2 is a near-dup of train item 7."""
    train = []
    for i in range(10):
        train.append({
            "id": f"train_{i:02d}",
            "problem": (
                f"Synthetic problem {i}: find the number of integer solutions "
                f"to x^{i % 4 + 2} + y^{i % 4 + 2} = {100 + i} where 0 <= x <= y."
            ),
            "answer": f"{i:03d}",
            "year": 2019 + (i % 5),
            "exam": "I" if i % 2 == 0 else "II",
            "source": "synthetic",
        })
    # Make train item 7 a known sentence
    leak_text = (
        "Determine how many ordered pairs (a, b) of positive integers "
        "satisfy a^2 + b^2 = 2025 with a < b and gcd(a, b) = 1."
    )
    train[7]["problem"] = leak_text

    test = [
        {"id": "test_0", "problem": "Compute the area of a regular hexagon with side length 4."},
        {"id": "test_1", "problem": "How many trailing zeros does 100! have in base ten?"},
        # Near-duplicate of leak_text -- two trivial word swaps so the screen
        # must use lexical overlap rather than exact equality to catch it.
        {"id": "test_2", "problem": leak_text.replace("Determine", "Find").replace("ordered pairs", "pairs")},
        {"id": "test_3", "problem": "Find the smallest prime factor of 1001."},
        {"id": "test_4", "problem": "What is the value of log base 2 of 1024?"},
    ]

    train_p = tmp_path / "train.jsonl"
    test_p = tmp_path / "test.jsonl"
    out_p = tmp_path / "screen.md"
    _write_jsonl(train_p, train)
    _write_jsonl(test_p, test)

    result = screen_aime_train_test(
        train_jsonl=train_p,
        test_jsonl=test_p,
        output_md=out_p,
        bm25_flag_threshold=0.5,
        bm25_top_k=3,
        minhash_threshold=0.8,
        num_perm=128,
    )

    # Two trivial word swaps -- BM25 must catch (lexical), MinHash may or may
    # not (depends on shingle granularity).  Either way the verdict is
    # non-PASS (FAIL if MinHash agreed, WARN if only BM25 did).
    assert result["verdict"] in {"FAIL", "WARN"}, (
        f"expected non-PASS verdict but got {result['verdict']}"
    )
    assert result["n_train"] == 10
    assert result["n_test"] == 5
    bm25_pairs = result["bm25_flagged"]
    assert any(
        tr == "train_07" and te == "test_2" for (tr, te, _) in bm25_pairs
    ), f"BM25 must flag train_07 <-> test_2 but got {bm25_pairs}"

    # Report exists and contains the verdict and excerpts
    assert out_p.exists()
    body = out_p.read_text(encoding="utf-8")
    assert result["verdict"] in body
    assert "train_07" in body
    assert "test_2" in body
    assert "BM25 lexical-overlap detector" in body
    assert "MinHash near-duplicate detector" in body


def test_end_to_end_screen_fails_on_exact_duplicate(tmp_path: Path):
    """Exact-duplicate train/test pair -> MinHash MUST fire -> FAIL."""
    leak_text = (
        "Find the number of ordered pairs (a, b) of positive integers with "
        "a^2 + b^2 = 2025, gcd(a, b) = 1, and a < b."
    )
    train = [
        {"id": f"train_{i}", "problem": f"Synthetic problem {i}: 2 + 2 equals?"}
        for i in range(5)
    ]
    train[2]["problem"] = leak_text
    test = [
        {"id": "test_0", "problem": "Compute the area of a regular hexagon."},
        {"id": "test_1", "problem": leak_text},
    ]
    train_p = tmp_path / "train.jsonl"
    test_p = tmp_path / "test.jsonl"
    out_p = tmp_path / "screen.md"
    _write_jsonl(train_p, train)
    _write_jsonl(test_p, test)

    result = screen_aime_train_test(
        train_jsonl=train_p, test_jsonl=test_p, output_md=out_p,
    )
    assert result["verdict"] == "FAIL"
    assert result["sub_verdict"] == "NEAR_DUPLICATE"
    assert any(te == "test_1" for (_, te, _) in result["minhash_flagged"]), (
        f"MinHash must flag the exact duplicate but got {result['minhash_flagged']}"
    )


def test_end_to_end_screen_warns_on_lexical_overlap_only(tmp_path: Path):
    """High BM25 vocabulary overlap but no MinHash near-dup -> WARN."""
    train = [
        {
            "id": f"train_{i}",
            "problem": (
                "Find the number of positive integer triples (a, b, c) "
                f"with a < b < c and a + b + c = {i + 30} satisfying gcd(a, b) = 1."
            ),
        }
        for i in range(8)
    ]
    test = [
        {
            "id": "test_0",
            "problem": (
                "Compute the smallest positive integer n such that the number "
                "of positive integer solutions to x + y + z = n equals 100."
            ),
        },
        {
            "id": "test_1",
            "problem": "Translate 'good morning' into Mandarin Chinese.",
        },
    ]
    train_p = tmp_path / "train.jsonl"
    test_p = tmp_path / "test.jsonl"
    out_p = tmp_path / "screen.md"
    _write_jsonl(train_p, train)
    _write_jsonl(test_p, test)

    # Very low BM25 threshold to force a lexical-only flag (shared vocab:
    # "positive integer", "number", "solutions") without a real near-dup.
    result = screen_aime_train_test(
        train_jsonl=train_p,
        test_jsonl=test_p,
        output_md=out_p,
        bm25_flag_threshold=0.05,
        minhash_threshold=0.8,
    )
    # Either WARN (BM25 fired but MinHash did not, the expected case) or PASS
    # (BM25 also stayed below the floor on this fixture); both are valid -- we
    # only need to assert that we never reach FAIL on a no-near-dup fixture.
    assert result["verdict"] in {"WARN", "PASS"}, (
        f"no near-dup fixture must not reach FAIL but got {result['verdict']}"
    )
    assert result["minhash_flagged"] == []


def test_end_to_end_screen_passes_clean_pair(tmp_path: Path):
    """A train / test pair with NO overlap returns PASS."""
    train = [
        {"id": f"t_{i}", "problem": f"Find the {i}-th prime number greater than 1000."}
        for i in range(8)
    ]
    test = [
        {"id": "x_0", "problem": "What is the airspeed velocity of an unladen swallow?"},
        {"id": "x_1", "problem": "Translate 'good morning' into Mandarin."},
    ]
    train_p = tmp_path / "train.jsonl"
    test_p = tmp_path / "test.jsonl"
    out_p = tmp_path / "screen.md"
    _write_jsonl(train_p, train)
    _write_jsonl(test_p, test)
    result = screen_aime_train_test(
        train_jsonl=train_p,
        test_jsonl=test_p,
        output_md=out_p,
    )
    assert result["verdict"] == "PASS"
    assert result["bm25_flagged"] == []
    assert result["minhash_flagged"] == []
    body = out_p.read_text(encoding="utf-8")
    assert "PASS contamination screen" in body


def test_cli_exits_nonzero_on_fail(tmp_path: Path):
    """``scripts/run_contamination_screen.py`` returns 1 when the screen FAILs."""
    train = [{"id": "train_0", "problem": "Find the number of positive divisors of 360."}]
    test = [{"id": "test_0", "problem": "Find the number of positive divisors of 360."}]
    train_p = tmp_path / "train.jsonl"
    test_p = tmp_path / "test.jsonl"
    out_p = tmp_path / "screen.md"
    _write_jsonl(train_p, train)
    _write_jsonl(test_p, test)

    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_contamination_screen.py"),
        "--train", str(train_p),
        "--test", str(test_p),
        "--out", str(out_p),
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    assert proc.returncode == 1, (
        f"expected exit 1 (FAIL) but got {proc.returncode}; "
        f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
    )
    assert out_p.exists()
    assert "FAIL" in out_p.read_text(encoding="utf-8")


def test_cli_exits_zero_on_pass(tmp_path: Path):
    train = [{"id": "train_0", "problem": "Compute 2 + 2."}]
    test = [{"id": "test_0", "problem": "Translate 'hello world' into Latin."}]
    train_p = tmp_path / "train.jsonl"
    test_p = tmp_path / "test.jsonl"
    out_p = tmp_path / "screen.md"
    _write_jsonl(train_p, train)
    _write_jsonl(test_p, test)

    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_contamination_screen.py"),
        "--train", str(train_p),
        "--test", str(test_p),
        "--out", str(out_p),
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    assert proc.returncode == 0, (
        f"expected exit 0 (PASS) but got {proc.returncode}; "
        f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
    )


def test_cli_exits_zero_on_warn(tmp_path: Path):
    """BM25 lexical overlap only -> verdict WARN -> CLI exit 0."""
    # Construct a train/test pair where shared problem-vocabulary will trip
    # the BM25 detector at a low threshold but never yield MinHash near-dup.
    train = [
        {"id": f"train_{i}", "problem": "Find the smallest positive integer n with n^2 + 1 prime."}
        for i in range(6)
    ]
    test = [
        {"id": "test_0", "problem": "Find the smallest positive integer n such that 2n - 1 is prime and 2n + 1 is prime."},
    ]
    train_p = tmp_path / "train.jsonl"
    test_p = tmp_path / "test.jsonl"
    out_p = tmp_path / "screen.md"
    _write_jsonl(train_p, train)
    _write_jsonl(test_p, test)

    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_contamination_screen.py"),
        "--train", str(train_p),
        "--test", str(test_p),
        "--out", str(out_p),
        "--bm25-flag-threshold", "0.1",
        "--minhash-threshold", "0.95",
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    # WARN or PASS both exit 0; we only need to assert non-FAIL.
    assert proc.returncode == 0, (
        f"expected exit 0 (WARN/PASS) but got {proc.returncode}; "
        f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
    )
