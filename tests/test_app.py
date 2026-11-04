from pathlib import Path

import pytest

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest


def test_app_runs_with_and_without_csa():
    at = AppTest.from_file(str(Path(__file__).parents[1] / "app.py"), default_timeout=120).run()
    assert not at.exception
    assert at.metric[0].label == "CVA"
    at.toggle[0].set_value(False).run()
    assert not at.exception and "without CSA" in at.subheader[0].value
