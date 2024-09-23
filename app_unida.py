import streamlit as st
import pandas as pd
import numpy as np
import pickle
import tempfile
from pycaret.regression import setup, compare_models, create_model, tune_model, plot_model, evaluate_model, finalize_model, predict_model, save_model, load_model

with open('best_model.pkl', 'rb') as model_file:
    modelo = pickle.load(model_file)
        
# Cargar los datos de prueba
prueba = pd.read_csv("prueba_APP.csv")
dataset = pd.read_csv( "dataset_APP.csv") 

# Imprimir las columnas para verificar

# Convertir las columnas categóricas en objetos si existen
ct = ["dominio", "Tec"]
for k in ct:
    if k in prueba.columns:
        prueba[k] = prueba[k].astype("O")

if 'test_data' not in st.session_state:
    st.session_state['test_data'] = pd.read_csv('dataset_APP.csv')

def prediccion_individual():
    st.header("Predicción manual de datos")
    
    dominio = st.selectbox("dominio", ['yahoo', 'Otro', 'gmail', 'hotmail'])
    Tec = st.selectbox("Tec", ['PC', 'Smartphone', 'Iphone', 'Portatil'])
    Avg_Session_Length = st.text_input("Avg. Session Length", value="32.06")
    Time_on_App = st.text_input("Time on App", value="10.7")
    Time_on_Website = st.text_input("Time on Website", value="37.71")
    Length_of_Membership = st.text_input("Length of Membership", value="3.004")

    if st.button("Calcular predicción manual"):
    
            Avg_Session_Length = float(Avg_Session_Length)
            Time_on_App = float(Time_on_App)
            Time_on_Website = float(Time_on_Website)
            Length_of_Membership = float(Length_of_Membership)
            
            # Crear el dataframe a partir de los inputs del usuario
            user = pd.DataFrame({
                'dominio': [dominio],
                'Tec': [Tec],
                'Avg. Session Length': [Avg_Session_Length],
                'Time on App': [Time_on_App],
                'Time on Website': [Time_on_Website],
                'Length of Membership': [Length_of_Membership],
                
            })

            # Hacer predicciones utilizando el modelo cargado
            predictions = predict_model(modelo, data=user)

            # Mostrar la predicción al usuario
            st.write(f'La predicción es: {predictions["prediction_label"][0]}')

    if st.button("Volver al menú principal"):
            st.session_state['menu'] = 'main'

def prediccion_base_datos():
    st.header("Cargar archivo para predecir")

    uploaded_file = st.file_uploader("Cargar archivo Excel", type=["xlsx", "csv"])

    if st.button("Predecir"):
        if uploaded_file is not None:
            try:
                # Cargar el archivo subido
                with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_path = tmp_file.name
                
                if uploaded_file.name.endswith(".csv"):
                    prueba = pd.read_csv(tmp_path)
                else:
                    prueba = pd.read_excel(tmp_path)

                cuantitativas = ['Avg. Session Length','Time on App','Time on Website','Length of Membership']
                categoricas = ['dominio', 'Tec']
                

                base_modelo = pd.concat([prueba.get(cuantitativas),prueba.get(categoricas)],axis = 1)

                # Realizar predicción
                df_test = base_modelo.copy()
                predictions = predict_model(modelo, data=df_test)
                predictions["price"] = predictions["prediction_label"]

            
                    # Preparar archivo para descargar
                kaggle = pd.DataFrame({'email': prueba["Email"], 'price': predictions["price"]})

                # Mostrar predicciones en pantalla
                st.write("Predicciones generadas correctamente!")
                st.write(kaggle)

                # Botón para descargar el archivo de predicciones
                st.download_button(label="Descargar archivo de predicciones",
                                data=kaggle.to_csv(index=False),
                                file_name="kaggle_predictions.csv",
                                mime="text/csv")

            except Exception as e:
                st.error(f"Error: {e}")
        else:
            st.error("Por favor, cargue un archivo válido.")

    if st.button("Volver al menú principal"):
        st.session_state['menu'] = 'main'

# Función principal para mostrar el menú de opciones
def menu_principal():
    st.title("API de Predicción Académica")
    option = st.selectbox("Seleccione una opción", ["", "Predicción Individual", "Predicción Base de Datos"])

    if option == "Predicción Individual":
        st.session_state['menu'] = 'individual'
    elif option == "Predicción Base de Datos":
        st.session_state['menu'] = 'base_datos'

# Lógica para manejar el flujo de la aplicación
if 'menu' not in st.session_state:
    st.session_state['menu'] = 'main'

if st.session_state['menu'] == 'main':
    menu_principal()
elif st.session_state['menu'] == 'individual':
    prediccion_individual()
elif st.session_state['menu'] == 'base_datos':
    prediccion_base_datos()