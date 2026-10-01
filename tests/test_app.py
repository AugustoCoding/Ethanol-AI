"""Teste de fumaça do app: as três abas rodam sem erro e mostram resultados."""
from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parent.parent / "streamlit_app.py")


def test_app_runs_and_shows_results():
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception, [e.value for e in at.exception]
    labels = [m.label for m in at.metric]
    for expected in ("Hemicellulose solubilized", "Glucose yield", "Minimum enzyme", "Temperature"):
        assert expected in labels


def test_extrapolation_is_flagged():
    at = AppTest.from_file(APP, default_timeout=120).run()
    at.slider(key="pre_temperature").set_value(216.0).run()
    assert not at.exception
    html = " ".join(h.proto.body for h in at.get("html"))
    assert "Temperature outside the experimentally validated range" in html


def test_bagasse_shows_empty_state():
    at = AppTest.from_file(APP, default_timeout=120).run()
    at.selectbox(key="hyd_biomass").set_value("Sugarcane Bagasse").run()
    assert not at.exception
    html = " ".join(h.proto.body for h in at.get("html"))
    assert "Model under development" in html
