"""Own parameters of the synthetic initial margin, calibrated on the public ECB history (market/im_params.csv).

    python scripts/calibrate_im.py

Not ISDA SIMM parameters. Risk weights: 99% quantile of absolute 10-business-day moves of ECB AAA zero rates, on the
recent three years with the worst one-year stress window. Correlations: of the same moves.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))
from xva.initial_margin import PARAMS, calibrate, params_table  # noqa: E402
from xva.validation import full_history  # noqa: E402

c = calibrate(full_history())
head = (f"# Own calibration, not ISDA SIMM parameters. ECB AAA zero rates, 10-business-day moves in bp, {c['first']:%Y-%m-%d} to "
        f"{c['last']:%Y-%m-%d}. Sample: last three years and the one-year stress window ending {c['stress_end']:%Y-%m-%d}.\n")
PARAMS.write_text(head + params_table(c["rw"], c["rho"]).to_csv(index=False, lineterminator="\n"), encoding="utf-8")
print(head.strip())
print(params_table(c["rw"], c["rho"])[["vertex", "risk_weight_bp"]].to_string(index=False))
print("min rho", c["rho"].min())
