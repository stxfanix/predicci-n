import streamlit as st
import joblib

# Cargar modelo y transformador
modelo = joblib.load('modelo_entrenado.joblib')
transformador = joblib.load('transformador_poly.joblib')

# Título de la aplicación
st.title("Aplicación para predicción del índice de masa corporal (IMC)")

# Crear las entradas del usuario
edad = st.slider("Selecciona tu edad:", min_value=1, max_value=100, value=25)
glucosa = st.slider("Selecciona tu nivel de glucosa:", min_value=1, max_value=300, value=100)

hipertension = st.radio(
    "¿Usted padece hipertensión?",
    options=["Sí, tengo hipertensión", "No, no tengo hipertensión"]
)

cardiopatia = st.radio(
    "¿Usted tiene alguna cardiopatía?",
    options=["Sí, padezco de cardiopatía", "No, no padezco de cardiopatías"]
)

acv = st.radio(
    "¿Ha sufrido un accidente cerebrovascular (ACV)?",
    options=["Sí, he sufrido un ACV", "No, no he sufrido un ACV"]
)

opciones_fumar = ["Never smoked", "Formerly smoked", "Smokes", "No smokes"]
humo = st.selectbox("Seleccione su hábito de consumo de tabaco:", opciones_fumar)

# Convertir las entradas a valores numéricos
hipertension_valor = 1 if hipertension == "Sí, tengo hipertensión" else 0
cardiopatia_valor = 1 if cardiopatia == "Sí, padezco de cardiopatía" else 0
acv_valor = 1 if acv == "Sí, he sufrido un ACV" else 0

humo_valor = {
    "Never smoked": 0,
    "Formerly smoked": 1,
    "Smokes": 2,
    "No smokes": 3
}[humo]

# Preparar las características de entrada
caracteristicas = [
    float(edad),
    float(glucosa),
    hipertension_valor,
    cardiopatia_valor,
    acv_valor,
    humo_valor
]

# Botón para realizar la predicción
if st.button("Realizar predicción"):
    # Transformar las características usando el transformador polinómico
    caracteristicas_poly = transformador.transform([caracteristicas])

    # Realizar predicción con el modelo
    prediccion_imc = modelo.predict(caracteristicas_poly)[0]

    # Mostrar el resultado de la predicción
    st.write(f"El IMC predicho es: {prediccion_imc:.2f}")

    # Clasificación del IMC
    if prediccion_imc < 18.5:
        st.write("Clasificación: Peso insuficiente.")
    elif 18.5 <= prediccion_imc < 24.9:
        st.write("Clasificación: Saludable.")
    elif 25 <= prediccion_imc < 29.9:
        st.write("Clasificación: Sobrepeso.")
    else:
        st.write("Clasificación: Obesidad.")
