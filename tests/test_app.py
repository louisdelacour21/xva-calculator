from pathlib import Path

import pytest

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest


def test_app_tourne_avec_et_sans_csa():
    at = AppTest.from_file(str(Path(__file__).parents[1] / "app.py"), default_timeout=120).run()
    assert not at.exception
    assert at.metric[0].label == "CVA"
    at.toggle[0].set_value(False).run()
    assert not at.exception and "sans CSA" in at.subheader[0].value
