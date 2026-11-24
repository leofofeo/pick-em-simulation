from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_app_defaults_and_invalid_input():
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "streamlit_app.py"), default_timeout=60).run()
    assert not app.exception
    assert len(app.metric) == 4
    assert len(app.dataframe) >= 4
    app.text_input[0].set_value("invalid")
    app.button[0].click().run()
    assert not app.exception
    assert app.error


def test_app_runs_weekly_without_standings():
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "streamlit_app.py"), default_timeout=60).run()
    app.number_input[1].set_value(5)
    app.number_input[2].set_value(4)
    app.number_input[5].set_value(1000)
    app.selectbox[0].select("None")
    app.text_input[0].set_value("2,3")
    app.text_input[1].set_value("2,4")
    app.button[0].click().run()
    assert not app.exception
    assert not app.error
    assert any("Weekly progression" in heading.value for heading in app.header)
