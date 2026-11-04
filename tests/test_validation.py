import numpy as np
from scipy import stats

from xva.hw1f import HW1F
from xva.market import sample_market
from xva.products import swap
from xva.validation import convergence, ks_power, pit
from xva.xva import load_credit

MKT = sample_market()


def test_pit_uniform_if_model_is_true():
    """Realised curves drawn from the model itself: the PITs must be uniform."""
    h, m = 0.25, HW1F(0.03, 0.008, MKT.estr)
    x = m.simulate([h], 200, seed=7, antithetic=False)["x"][:, 0]
    pits = [pit(MKT.estr, lambda T, xi=xi: m.P(h, h + np.asarray(T, dtype=float), xi), h, 0.008, n=1000, seed=i)
            for i, xi in enumerate(x)]
    for p in pits[0]:
        assert stats.kstest([q[p] for q in pits], "uniform").pvalue > 0.01


def test_ks_power():
    assert abs(ks_power(50, factor=1.0) - 0.05) < 0.02  # with no misfit, KS rejects at the nominal level
    assert ks_power(20) < ks_power(250)


def test_square_root_n_convergence():
    tr = [swap("P", "irs", 1, 10e6, 0.025, 0.0, 10)]
    df, slope = convergence(tr, HW1F(0.03, 0.008, MKT.estr), MKT, load_credit(P=MKT.estr), (1_000, 4_000, 16_000))
    assert abs(slope + 0.5) < 0.1 and (df["antithetic gain"] > 1).all()
