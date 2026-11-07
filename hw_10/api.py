from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel, Field

from predict import load_artifacts, predict_price

app = FastAPI(
    title="Realty Price Prediction API",
    description="Прогноз стоимости недвижимости по площади, числу комнат и городу",
    version="1.0.0",
)


# ---------- Pydantic-модели ----------

class PredictionRequest(BaseModel):
    total_square: float = Field(..., gt=0, description="Площадь, м²")
    rooms: float = Field(..., ge=0, description="Количество комнат")
    city: str = Field(..., description="Город")


class PredictionResponse(BaseModel):
    predicted_price: float = Field(..., description="Прогноз стоимости, руб.")
    price_per_sqm: float = Field(..., description="Цена за м², руб.")


# ---------- 1. Health-check ----------

@app.get("/health", tags=["health"])
def health():
    """Liveness-проба: сервис жив и модель загружена."""
    try:
        load_artifacts()
        return {"status": "ok", "model_loaded": True}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"model not loaded: {e}")


# ---------- 2. GET-предсказание ----------

@app.get("/predict_get", response_model=PredictionResponse, tags=["predict"])
def predict_get(
    total_square: float = Query(..., gt=0, description="Площадь, м²"),
    rooms: float = Query(..., ge=0, description="Количество комнат"),
    city: str = Query(..., description="Город"),
):
    """Предсказание через GET. Параметры передаются в query string."""
    try:
        price = predict_price(total_square, rooms, city)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"prediction error: {e}")
    return PredictionResponse(
        predicted_price=price,
        price_per_sqm=price / total_square,
    )


# ---------- 3. POST-предсказание ----------

@app.post("/predict_post", response_model=PredictionResponse, tags=["predict"])
def predict_post(payload: PredictionRequest):
    """Предсказание через POST. Параметры передаются JSON-телом."""
    try:
        price = predict_price(payload.total_square, payload.rooms, payload.city)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"prediction error: {e}")
    return PredictionResponse(
        predicted_price=price,
        price_per_sqm=price / payload.total_square,
    )