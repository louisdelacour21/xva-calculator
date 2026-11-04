import numpy as np
import pytest

from xva.exposure import exposure, grid
from xva.hw1f import HW1F
from xva.market import sample_market, swaption_vols
from xva.products import cube_mtm, swap
from xva.risk import PILLARS, ba_cva_reduced, cs01, duration_factor, sa_ccr, scva, sensitivities, triangle
from xva.xva import load_credit

MKT = sample_market()
CREDIT = load_credit(P=MKT.estr)
PAYER = swap("P", "irs", 1, 10e6, 0.025, 0.0, 10)


def test_triangles_sum_to_one():
    u = np.linspace(0, 40, 401)
    assert np.allclose(sum(triangle(k)(u) for k in range(len(PILLARS))), 1.0)


@pytest.fixture(scope="module")
def sens():
    return sensitivities([PAYER], MKT, swaption_vols(), CREDIT, n=2000)


def test_pillar_deltas_sum_to_parallel(sens):
    for q in ("CVA", "DVA"):
        total = sum(sens[f"delta {p:g}Y"][q][0] for p in PILLARS)
        assert abs(total / sens["delta parallel"][q][0] - 1) < 0.05


def test_delta_vega_signs(sens):
    """Payer: rates up → more positive exposure (CVA ↑) and less negative (DVA ↓); vol ↑ → both ↑."""
    d, v = sens["delta parallel"], sens["vega +1%"]
    assert d["CVA"][0] > 5 * d["CVA"][1] and d["DVA"][0] < 0
    assert v["CVA"][0] > 5 * v["CVA"][1] and v["DVA"][0] > 0


def test_cs01():
    hw = HW1F(0.03, 0.009, MKT.estr)
    base, t = grid([PAYER], mpors=())
    sim = hw.simulate(t, 2000)
    expo = exposure(cube_mtm([PAYER], hw, MKT, sim).sum(-1), sim["D"], t, base, 0.0)
    r = cs01(expo, CREDIT, MKT.estr)
    total = sum(r[f"CS01 {T:g}Y"]["CVA"][0] for T in CREDIT.counterparty.spreads)
    assert r["CS01 parallel"]["CVA"][0] > 0 and abs(total / r["CS01 parallel"]["CVA"][0] - 1) < 0.05
    assert r["CS01 parallel"]["DVA"][0] < 0  # DVA only counts if the counterparty survives


# --- capital, computed by hand -----------------------------------------------------------------------------------

def test_sa_ccr_single_unmargined_trade():
    # payer 10M, 0-5 years: SD = (1 - e^-0.25)/0.05 = 4.42398; MF = 1; add-on = 0.5%·44.2398M = 221,199
    r = sa_ccr([swap("A", "irs", 1, 10e6, 0.025, 0.0, 5)], V0=0.0)
    assert duration_factor(0, 5) == pytest.approx(4.423984, abs=1e-6)
    assert r["add-on"] == pytest.approx(221_199.2, abs=0.1)
    assert r["multiplier"] == 1.0 and r["EAD"] == pytest.approx(1.4 * 221_199.2, abs=0.2)  # 309,679


def test_sa_ccr_two_buckets_and_multiplier():
    # payer 10M 5 years (bucket 1-5) + receiver 10M 10 years (bucket > 5): D2 = 44.2398M, D3 = -78.6939M
    tr = [swap("A", "irs", 1, 10e6, 0.025, 0.0, 5), swap("B", "irs", -1, 10e6, 0.025, 0.0, 10)]
    en = np.sqrt(44.23984**2 + 78.69387**2 - 1.4 * 44.23984 * 78.69387) * 1e6  # 57.24M
    V0 = -100_000.0
    a = 0.005 * en
    mult = 0.05 + 0.95 * np.exp(V0 / (2 * 0.95 * a))
    r = sa_ccr(tr, V0)
    assert r["add-on"] == pytest.approx(a, rel=1e-6) and r["RC"] == 0.0
    assert r["EAD"] == pytest.approx(1.4 * mult * a, rel=1e-6)


def test_sa_ccr_margined():
    # CSA without threshold, MPoR 10 d: MF = 1.5·√(10/250) = 0.3; RC = max(V - C, TH + MTA, 0) = 0
    csa = {"H_C": 0.0, "H_B": 0.0, "MTA": 0.0, "mpor_days": 10}
    r = sa_ccr([swap("A", "irs", 1, 10e6, 0.025, 0.0, 5)], V0=0.0, csa=csa)
    assert r["add-on"] == pytest.approx(0.3 * 221_199.2, abs=0.1)
    assert r["EAD"] == pytest.approx(1.4 * 0.3 * 221_199.2, abs=0.1) and r["EAD"] < r["unmargined EAD"]
    # huge threshold: the margined EAD would be larger, capped at the unmargined one
    assert sa_ccr([swap("A", "irs", 1, 10e6, 0.025, 0.0, 5)], 0.0, csa | {"H_C": 1e9})["EAD"] == r["unmargined EAD"]


def test_ba_cva_reduced():
    # RW 3%, M = 5, EAD = 309,679: DF = (1 - e^-0.25)/0.25 = 0.884797; SCVA = 0.03·5·309,679·0.884797/1.4 = 29,357
    s = scva(0.03, [(5.0, 309_679.0)])
    assert s == pytest.approx(0.03 * 5 * 309_679.0 * 0.884797 / 1.4, rel=1e-6)
    assert ba_cva_reduced([s]) == pytest.approx(0.65 * s, rel=1e-12)  # one counterparty: √(0.25 + 0.75) = 1
    assert ba_cva_reduced([s, s]) == pytest.approx(0.65 * s * np.sqrt(2.5), rel=1e-12)  # √(0.25·4 + 0.75·2)
