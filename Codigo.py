import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta

import tkinter as tk
from tkinter import ttk
from tkinter import messagebox

# --- Red neuronal para determinar el riesgo en el embarazo ---

def entrenar_red_neuronal(data):
  """Entrena la red neuronal para determinar el riesgo en el embarazo."""

  # 1. Preprocesamiento de datos
  #   2.1 Convertir variables categóricas a numéricas
  data['hipertension_arterial'] = data['hipertension_arterial'].map({'Si': 1, 'No': 0})
  data['diabetes'] = data['diabetes'].map({'Si': 1, 'No': 0})
  data['preeclampsia'] = data['preeclampsia'].map({'Si': 1, 'No': 0})
  data['hipertension_cronica'] = data['hipertension_cronica'].map({'Si': 1, 'No': 0})
  data['sobrepeso'] = data['sobrepeso'].map({'Si': 1, 'No': 0})

  #   2.2 Separar datos en características (X) y etiquetas (y)
  X = data[['hipertension_arterial', 'diabetes', 'preeclampsia', 'hipertension_cronica', 'sobrepeso']]
  y = data['riesgo_embarazo']

  #   2.3 Dividir datos en conjuntos de entrenamiento y prueba
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

  #   2.4 Escalar los datos (opcional, pero recomendado para redes neuronales)
  scaler = StandardScaler()
  X_train = scaler.fit_transform(X_train)
  X_test = scaler.transform(X_test)

  # 3. Construir la red neuronal
  model = Sequential()
  model.add(Dense(8, activation='relu', input_shape=(X_train.shape[1],)))  # Capa oculta con 8 neuronas
  model.add(Dense(1, activation='sigmoid'))  # Capa de salida con 1 neurona (probabilidad de riesgo)

  # 4. Compilar el modelo
  model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

  # 5. Entrenar el modelo
  history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))  # Ajusta epochs y batch_size según sea necesario

  # 6. Evaluar el modelo
  _, accuracy = model.evaluate(X_test, y_test)
  print('Accuracy: %.2f' % (accuracy * 100))

  return model, scaler

# --- Regresión lineal para predecir la fecha de parto ---

def predecir_fecha_parto(ultima_menstruacion, semanas_embarazo=0, dias_embarazo=0):
  """
    Predicción de la fecha de parto utilizando regresión lineal.
  """

  fecha_ultima_menstruacion = datetime.strptime(ultima_menstruacion, "%Y-%m-%d")

  data = pd.DataFrame({
      'dias_desde_ultima_menstruacion': [0], 
      'semanas_embarazo': [semanas_embarazo],
      'dias_embarazo': [dias_embarazo]
  })

  model = LinearRegression()

  X_train = pd.DataFrame({
      'dias_desde_ultima_menstruacion': [0, 280, 280 * 7], 
      'semanas_embarazo': [0, 40, 40],
      'dias_embarazo': [0, 280, 280 * 7]
  })
  y_train = [fecha_ultima_menstruacion, fecha_ultima_menstruacion + timedelta(days=280), 
              fecha_ultima_menstruacion + timedelta(weeks=40)]
  model.fit(X_train, y_train)

  fecha_parto_estimada = model.predict(data)[0]
  fecha_parto_estimada = fecha_parto_estimada.strftime("%Y-%m-%d")

  return fecha_parto_estimada

# --- Interfaz gráfica ---

class Aplicacion(tk.Tk):
    def __init__(self):
        super().__init__()

        # Configuración de la ventana
        self.title("Calculadora de Embarazo")
        self.geometry("500x400")
        self.configure(bg="#87CEEB")  # Color azul celeste

        # Crear la etiqueta de título
        label_titulo = tk.Label(self, text="Calculadora de Embarazo", font=("Arial", 16), bg="#87CEEB")
        label_titulo.pack(pady=10)

        # Crear el marco para los datos del paciente
        frame_datos = tk.Frame(self, bg="#87CEEB")
        frame_datos.pack(pady=10)

        # --- Datos del paciente ---

        # Etiqueta y campo de entrada para la última menstruación
        label_ultima_menstruacion = tk.Label(frame_datos, text="Última menstruación (YYYY-MM-DD):", bg="#87CEEB")
        label_ultima_menstruacion.grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.entry_ultima_menstruacion = tk.Entry(frame_datos)
        self.entry_ultima_menstruacion.grid(row=0, column=1, padx=5, pady=5)

        # Etiqueta y campo de entrada para las semanas de embarazo
        label_semanas_embarazo = tk.Label(frame_datos, text="Semanas de embarazo:", bg="#87CEEB")
        label_semanas_embarazo.grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.entry_semanas_embarazo = tk.Entry(frame_datos)
        self.entry_semanas_embarazo.grid(row=1, column=1, padx=5, pady=5)

        # Etiqueta y campo de entrada para los días de embarazo
        label_dias_embarazo = tk.Label(frame_datos, text="Días de embarazo:", bg="#87CEEB")
        label_dias_embarazo.grid(row=2, column=0, padx=5, pady=5, sticky="w")
        self.entry_dias_embarazo = tk.Entry(frame_datos)
        self.entry_dias_embarazo.grid(row=2, column=1, padx=5, pady=5)

        # --- Datos para el riesgo ---

        # Etiqueta y casilla de verificación para hipertensión arterial
        self.var_hipertension_arterial = tk.IntVar()
        check_hipertension_arterial = tk.Checkbutton(frame_datos, text="Hipertensión arterial", variable=self.var_hipertension_arterial, bg="#87CEEB")
        check_hipertension_arterial.grid(row=3, column=0, columnspan=2, padx=5, pady=5, sticky="w")

        # Etiqueta y casilla de verificación para diabetes
        self.var_diabetes = tk.IntVar()
        check_diabetes = tk.Checkbutton(frame_datos, text="Diabetes", variable=self.var_diabetes, bg="#87CEEB")
        check_diabetes.grid(row=4, column=0, columnspan=2, padx=5, pady=5, sticky="w")

        # Etiqueta y casilla de verificación para preeclampsia
        self.var_preeclampsia = tk.IntVar()
        check_preeclampsia = tk.Checkbutton(frame_datos, text="Preeclampsia", variable=self.var_preeclampsia, bg="#87CEEB")
        check_preeclampsia.grid(row=5, column=0, columnspan=2, padx=5, pady=5, sticky="w")

        # Etiqueta y casilla de verificación para hipertensión crónica
        self.var_hipertension_cronica = tk.IntVar()
        check_hipertension_cronica = tk.Checkbutton(frame_datos, text="Hipertensión crónica", variable=self.var_hipertension_cronica, bg="#87CEEB")
        check_hipertension_cronica.grid(row=6, column=0, columnspan=2, padx=5, pady=5, sticky="w")

        # Etiqueta y casilla de verificación para sobrepeso
        self.var_sobrepeso = tk.IntVar()
        check_sobrepeso = tk.Checkbutton(frame_datos, text="Sobrepeso", variable=self.var_sobrepeso, bg="#87CEEB")
        check_sobrepeso.grid(row=7, column=0, columnspan=2, padx=5, pady=5, sticky="w")

        # --- Resultados ---

        # Etiqueta para mostrar la fecha de parto
        self.label_fecha_parto = tk.Label(self, text="Fecha de parto estimada:", bg="#87CEEB")
        self.label_fecha_parto.pack(pady=5)

        # Etiqueta para mostrar el riesgo de embarazo
        self.label_riesgo_embarazo = tk.Label(self, text="Riesgo de embarazo:", bg="#87CEEB")
        self.label_riesgo_embarazo.pack(pady=5)

        # Botón para calcular
        boton_calcular = tk.Button(self, text="Calcular", command=self.calcular, bg="#4CAF50", fg="white", font=("Arial", 12))
        boton_calcular.pack(pady=10)

        # Iniciar el bucle de la ventana
        self.mainloop()

    def calcular(self):
        """Calcula la fecha de parto y el riesgo del embarazo."""

        # Obtener datos del usuario
        ultima_menstruacion = self.entry_ultima_menstruacion.get()
        semanas_embarazo = int(self.entry_semanas_embarazo.get()) if self.entry_semanas_embarazo.get() else 0
        dias_embarazo = int(self.entry_dias_embarazo.get()) if self.entry_dias_embarazo.get() else 0

        # --- Validar los datos ---

        # Validar la fecha de la última menstruación
        try:
            datetime.strptime(ultima_menstruacion, "%Y-%m-%d")
        except ValueError:
            messagebox.showerror("Error", "Formato de fecha inválido. Ingrese la fecha en formato YYYY-MM-DD.")
            return

        # Validar que las semanas y días de embarazo sean números
        try:
            semanas_embarazo = int(semanas_embarazo)
            dias_embarazo = int(dias_embarazo)
        except ValueError:
            messagebox.showerror("Error", "Las semanas y días de embarazo deben ser números enteros.")
            return

        # --- Predecir la fecha de parto ---

        fecha_parto_estimada = predecir_fecha_parto(ultima_menstruacion, semanas_embarazo, dias_embarazo)

        # --- Evaluar el riesgo del embarazo ---

        # Crear un DataFrame con los datos del usuario
        data = pd.DataFrame({
            'hipertension_arterial': [self.var_hipertension_arterial.get()],
            'diabetes': [self.var_diabetes.get()],
            'preeclampsia': [self.var_preeclampsia.get()],
            'hipertension_cronica': [self.var_hipertension_cronica.get()],
            'sobrepeso': [self.var_sobrepeso.get()]
        })

        # Entrenar la red neuronal
        model, scaler = entrenar_red_neuronal(data)

        # Predecir el riesgo
        X_pred = scaler.transform(data)
        riesgo_predicho = model.predict(X_pred)[0][0]

        # Mostrar resultados
        self.label_fecha_parto.config(text=f"Fecha de parto estimada: {fecha_parto_estimada}")
        self.label_riesgo_embarazo.config(text=f"Riesgo de embarazo: {'Alto' if riesgo_predicho > 0.5 else 'Bajo'}")

# Iniciar la aplicación
app = Aplicacion()
