import zipfile
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# ---------- 1. Чтение xlsx из ZIP ----------
ZIP_PATH = "train_data.zip"
SHEET = "realty_data"

with zipfile.ZipFile(ZIP_PATH) as z:
    xlsx_name = next(
        n for n in z.namelist()
        if n.lower().endswith(".xlsx") and not n.startswith("__MACOSX")
    )
    print(f"Читаю из архива: {xlsx_name}")
    with z.open(xlsx_name) as f:
        df = pd.read_excel(f, sheet_name=SHEET)

# Оставляем только нужные столбцы
cols = ["price", "total_square", "rooms", "city"]
df = df[cols].copy()

# ---------- 2. Предобработка ----------
# price и total_square — числовые
df["price"] = pd.to_numeric(df["price"], errors="coerce")
df["total_square"] = pd.to_numeric(df["total_square"], errors="coerce")

# rooms — приведём к числам (пустые -> NaN)
df["rooms"] = pd.to_numeric(df["rooms"], errors="coerce")

# city — нормализация текста
df["city"] = (
    df["city"]
    .astype(str)
    .str.strip()
    .replace({"": np.nan, "nan": np.nan, "None": np.nan})
)

# Отбрасываем строки без целевой переменной и площади
df = df.dropna(subset=["price", "total_square"])
df = df[df["total_square"] > 0]

# Убираем грубые выбросы по цене (1% и 99% квантили) — опционально
lo, hi = df["price"].quantile([0.01, 0.99])
df = df[(df["price"] >= lo) & (df["price"] <= hi)]

print(f"Строк после очистки: {len(df)}")
print(df.head())

# ---------- 3. Признаки и цель ----------
X = df[["total_square", "rooms", "city"]]
y = df["price"]

num_features = ["total_square", "rooms"]
cat_features = ["city"]

num_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

cat_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer([
    ("num", num_pipe, num_features),
    ("cat", cat_pipe, cat_features),
])

# ---------- 4. Обучение ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = Pipeline([
    ("prep", preprocessor),
    ("reg", RandomForestRegressor(
        n_estimators=400, max_depth=None, random_state=42, n_jobs=-1
    )),
])

model.fit(X_train, y_train)

pred = model.predict(X_test)
print(f"MAE: {mean_absolute_error(y_test, pred):,.0f} руб.")
print(f"R²:  {r2_score(y_test, pred):.4f}")

# ---------- 5. Сохранение ----------
joblib.dump(model, "realty_model.joblib")

# Список городов для выпадающего списка в UI
cities = sorted(df["city"].dropna().unique().tolist())
joblib.dump(cities, "cities.joblib")

print("Модель сохранена: realty_model.joblib")
print("Города:", cities)