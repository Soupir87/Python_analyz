import streamlit as st
from predict import load_artifacts, predict_price

st.set_page_config(page_title="Прогноз стоимости недвижимости", page_icon="🏠")
st.title("🏠 Прогноз стоимости недвижимости")
st.caption("Модель обучена на данных из train_data.zip (price, total_square, rooms, city)")

model, cities = load_artifacts()

with st.form("predict_form"):
    col1, col2 = st.columns(2)
    with col1:
        total_square = st.number_input(
            "Площадь, м²", min_value=5.0, max_value=500.0, value=50.0, step=0.5
        )
    with col2:
        rooms = st.number_input(
            "Количество комнат", min_value=0, max_value=10, value=2, step=1
        )

    city = st.selectbox("Город", cities)

    submitted = st.form_submit_button("Спрогнозировать стоимость")

if submitted:
    try:
        price = predict_price(total_square, rooms, city)
        st.success(f"### Прогнозируемая стоимость: **{price:,.0f} ₽**".replace(",", " "))
        st.metric("Цена за м²", f"{price / total_square:,.0f} ₽".replace(",", " "))
    except Exception as e:
        st.error(f"Ошибка прогноза: {e}")

with st.expander("О модели"):
    st.write(
        "Пайплайн: медианное заполнение пропусков + стандартизация числовых "
        "признаков, OneHot-кодирование города, RandomForestRegressor (400 деревьев)."
    )