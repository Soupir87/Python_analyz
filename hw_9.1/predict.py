import joblib
import pandas as pd

_model = None
_cities = None

def load_artifacts():
    global _model, _cities
    if _model is None:
        _model = joblib.load("realty_model.joblib")
        _cities = joblib.load("cities.joblib")
    return _model, _cities


def predict_price(total_square: float, rooms: float, city: str) -> float:
    model, _ = load_artifacts()
    X = pd.DataFrame([{
        "total_square": total_square,
        "rooms": rooms,
        "city": city,
    }])
    return float(model.predict(X)[0])