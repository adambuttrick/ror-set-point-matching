from ror_set_point.evaluation.stats import bootstrap, mcnemar_exact, mde_exact


def test_mcnemar_exact():
    assert mcnemar_exact(0, 0) == 1.0
    assert abs(mcnemar_exact(0, 10) - 2 * 0.5 ** 10) < 1e-12
    assert mcnemar_exact(3, 7) == mcnemar_exact(7, 3)
    assert mcnemar_exact(5, 5) == 1.0


def test_mde_falls_as_strings_increase():
    small, large = mde_exact(500, 0.1), mde_exact(5000, 0.1)
    assert 0 < large < small < 0.1


def test_bootstrap_interval_holds_the_estimate():
    per = [(1, 1, 1, 1)] * 80 + [(0, 0, 1, 1)] * 20
    iv = bootstrap(per, 300, 1)
    lo, hi = iv["exact"]
    assert lo <= 0.8 <= hi
    assert iv["precision"] == iv["recall"]
