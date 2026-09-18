"""UI tests for the Streamlit app.

Run from the project root:
    python -m unittest tests/test_app_ui.py -v
"""
import re
import sys
import unittest
from pathlib import Path

import joblib
import pandas as pd
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.features import MODEL_INPUTS  # noqa: E402

APP_PATH = str(ROOT / "app" / "app.py")
BUNDLE = joblib.load(ROOT / "models" / "gradient_boosting_model.joblib")


def run_app():
    return AppTest.from_file(APP_PATH, default_timeout=120).run()


def visible_text(at):
    """All text a user can see on the page."""
    parts = [el.value for el in at.title] + [el.value for el in at.subheader]
    parts += [el.value for el in at.markdown] + [el.value for el in at.caption]
    parts += [el.value for el in at.info] + [el.value for el in at.warning] + [el.value for el in at.error]
    parts += [el.label + " " + str(el.value) for el in at.metric]
    return "\n".join(str(p) for p in parts)


def checkbox(at, label_start):
    for box in at.checkbox:
        if box.label.startswith(label_start):
            return box
    raise AssertionError(f"checkbox not found: {label_start}")


def estimate(at, suburb, prop_type, car_unknown=False, land_unknown=False, building_unknown=True):
    at.selectbox(key="suburb").set_value(suburb).run()
    at.selectbox(key="ptype").set_value(prop_type).run()
    checkbox(at, "Car spaces unknown").set_value(car_unknown)
    if prop_type != "Unit":
        checkbox(at, "Land size unknown").set_value(land_unknown)
    checkbox(at, "Building / floor area unknown").set_value(building_unknown)
    at.button[0].click().run()
    return at


def shown_price(at):
    card = [m.value for m in at.markdown if 'class="result-card"' in m.value and "value" in m.value][-1]
    return int(re.search(r'class="value">\$([\d,]+)<', card).group(1).replace(",", ""))


class AppUITests(unittest.TestCase):

    def check_prediction(self, at):
        self.assertEqual(len(at.exception), 0, at.exception)
        result = at.session_state["result"]
        # The app must build exactly the columns the saved model expects.
        row = pd.DataFrame([result["values"]])[MODEL_INPUTS]
        for col in MODEL_INPUTS:
            if col not in ("suburb", "property_type"):
                row[col] = row[col].astype(float)
        expected = round(float(BUNDLE["model"].predict(row)[0]), -3)
        self.assertEqual(shown_price(at), expected)
        self.assertIn("Model estimate — not a professional property valuation.", visible_text(at))

    def test_initial_page_loads(self):
        at = run_app()
        self.assertEqual(len(at.exception), 0)
        self.assertEqual(at.title[0].value, "Sydney Housing Price Predictor")

    def test_model_name_and_training_count(self):
        text = visible_text(run_app())
        self.assertIn("Gradient Boosting Regressor", text)
        self.assertIn("105 sold properties", text)

    def test_blacktown_house(self):
        self.check_prediction(estimate(run_app(), "Blacktown", "House"))

    def test_parramatta_unit(self):
        self.check_prediction(estimate(run_app(), "Parramatta", "Unit"))

    def test_mosman_townhouse(self):
        self.check_prediction(estimate(run_app(), "Mosman", "Townhouse"))

    def test_missing_car_spaces(self):
        at = estimate(run_app(), "Blacktown", "House", car_unknown=True, building_unknown=False)
        self.check_prediction(at)
        self.assertTrue(pd.isna(at.session_state["result"]["values"]["car_spaces"]))

    def test_missing_land(self):
        at = estimate(run_app(), "Parramatta", "House", land_unknown=True, building_unknown=False)
        self.check_prediction(at)
        self.assertTrue(pd.isna(at.session_state["result"]["values"]["land_size_sqm"]))

    def test_missing_building_area(self):
        at = estimate(run_app(), "Mosman", "Unit", building_unknown=True)
        self.check_prediction(at)
        self.assertTrue(pd.isna(at.session_state["result"]["values"]["building_size_sqm"]))

    def test_mosman_house_warning(self):
        at = estimate(run_app(), "Mosman", "House")
        self.check_prediction(at)
        warnings = " ".join(w.value for w in at.warning)
        self.assertIn("Mosman houses were the hardest segment to predict", warnings)
        self.assertNotIn("_", warnings)  # no raw column names

    def test_no_internal_details_visible(self):
        at = estimate(run_app(), "Blacktown", "House")
        text = visible_text(at).lower()
        for bad in ["traceback", "debug", "/home/", "desktop/assingment", ".joblib", "sklearn", "exception"]:
            self.assertNotIn(bad, text)

    def test_no_ai_tool_wording(self):
        at = estimate(run_app(), "Mosman", "House")
        text = visible_text(at)
        for bad in ["LLM", "ChatGPT", "Claude", "AI-assisted", "Generative AI", "GenAI"]:
            self.assertNotIn(bad, text)
        self.assertIsNone(re.search(r"\bAI\b", text))


if __name__ == "__main__":
    unittest.main()
