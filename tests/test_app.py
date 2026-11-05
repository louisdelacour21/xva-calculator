from pathlib import Path

import pytest

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest


@pytest.fixture(scope="module")
def app():
    return AppTest.from_file(str(Path(__file__).parents[1] / "app.py"), default_timeout=240).run()


def metric(at, label):
    return next(m.value for m in at.metric if m.label == label)


def widget(items, label):
    return next(w for w in items if w.label == label)


def warnings(at):
    return " ".join(w.value for w in at.warning)


def test_first_load_runs_the_demo(app):
    assert not app.exception and not app.error
    assert [m.label for m in app.metric][:3] == ["Mark-to-market", "Total XVA", "Adjusted value"]
    assert metric(app, "Mark-to-market").endswith("M€")


def test_initial_margin_page_and_full_stack(app):
    assert metric(app, "Initial margin, central").endswith("M€")
    assert metric(app, "KVA") and metric(app, "XVA including MVA and KVA")
    assert any("not ISDA SIMM" in c.value for c in app.caption)
    assert widget(app.number_input, "Hurdle rate for KVA (%)").value == 10.0


def test_controls_baseline_and_log(app):
    assert metric(app, "Controls passed") == "8 of 8"
    assert app.session_state.run_log[-1]["run ID"] == metric(app, "Run ID")
    widget(app.button, "Pin this run as baseline").click().run()
    assert not app.exception and "baseline" in app.session_state
    assert any("None: same inputs." in c.value for c in app.code)
    widget(app.button, "Clear the baseline").click().run()
    assert not app.exception and "baseline" not in app.session_state


def test_inputs_are_typed_numbers_not_sliders(app):
    assert not app.slider
    assert widget(app.number_input, "Paths").value == 10_000


def test_a_changed_input_waits_for_run(app):
    cva = metric(app, "CVA")
    widget(app.number_input, "Paths").set_value(2000).run()
    assert not app.exception and "Press Run" in warnings(app)
    assert metric(app, "CVA") == cva  # previous results stay on screen
    widget(app.button, "Run").click().run()
    assert not app.exception and "Press Run" not in warnings(app)
    assert metric(app, "CVA") != cva


def test_csa_off_then_stress(app):
    app.toggle[0].set_value(False).run()
    widget(app.button, "Run").click().run()
    assert not app.exception and not app.error
    widget(app.button, "Run the stress test").click().run()
    assert not app.exception and not app.error


def test_sensitivities_on_demand(app):
    widget(app.number_input, "Paths for this table").set_value(200).run()
    widget(app.button, "Compute").click().run()
    assert not app.exception and not app.error


def test_curve_preset_then_reset(app):
    widget(app.selectbox, "Curve preset").set_value("Project 09 bootstrapped curves").run()
    widget(app.button, "Load").click().run()
    assert not app.exception and not app.error and "Press Run" not in warnings(app)
    assert any("Project 09" in c.value for c in app.caption)
    version = app.session_state.version
    widget(app.button, "Reset to the demo").click().run()
    assert not app.exception and widget(app.number_input, "Paths").value == 10_000
    assert app.session_state.version > version  # same keys: the browser would send its old cell edits back
    assert widget(app.selectbox, "Curve preset").value == "Flat demo curves"
