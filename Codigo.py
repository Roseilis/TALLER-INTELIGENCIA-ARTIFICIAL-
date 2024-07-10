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



    # Campos de entrada para los datos del paciente

    label_ultima_menstruacion = tk.Label(frame_datos, text="Última Menstruación (AAAA-MM-DD):", bg="#87CEEB")

    label_ultima_menstruacion.grid(row=0, column=0, padx=5, pady=5)

    self.entry_ultima_menstruacion = tk.Entry(frame_datos)

    self.entry_ultima_menstruacion.grid(row=0, column=1, padx=5, pady=5)



    label_semanas_embarazo = tk.Label(frame_datos, text="Semanas de Embarazo:", bg="#87CEEB")

    label_semanas_embarazo.grid(row=1, column=0, padx=5, pady=5)

    self.entry_semanas_embarazo = tk.Entry(frame_datos)

    self.entry_semanas_embarazo.grid(row=1, column=1, padx=5, pady=5)



    label_dias_embarazo = tk.Label(frame_datos, text="Días de Embarazo:", bg="#87CEEB")

    label_dias_embarazo.grid(row=2, column=0, padx=5, pady=5)

    self.entry_dias_embarazo = tk.Entry(frame_datos)

    self.entry_dias_embarazo.grid(row=2, column=1, padx=5, pady=5)



    # Marco para los antecedentes médicos

    self.frame_antecedentes = tk.Frame(self, bg="#87CEEB")

    self.frame_antecedentes.pack(pady=10)



    # Crear el checkbutton para antecedentes médicos

    self.check_hipertension_arterial = tk.Checkbutton(self.frame_antecedentes, text="Hipertensión Arterial", bg="#87CEEB")

    self.check_hipertension_arterial.grid(row=0, column=0, padx=5, pady=5, sticky="w")

    self.check_diabetes = tk.Checkbutton(self.frame_antecedentes, text="Diabetes", bg="#87CEEB")

    self.check_diabetes.grid(row=1, column=0, padx=5, pady=5, sticky="w")

    self.check_preeclampsia = tk.Checkbutton(self.frame_antecedentes, text="Preeclampsia", bg="#87CEEB")

    self.check_preeclampsia.grid(row=2, column=0, padx=5, pady=5, sticky="w")

    self.check_hipertension_cronica = tk.Checkbutton(self.frame_antecedentes, text="Hipertensión Crónica", bg="#87CEEB")

    self.check_hipertension_cronica.grid(row=3, column=0, padx=5, pady=5, sticky="w")

    self.check_sobrepeso = tk.Checkbutton(self.frame_antecedentes, text="Sobrepeso", bg="#87CEEB")

    self.check_sobrepeso.grid(row=4, column=0, padx=5, pady=5, sticky="w")



    # Botones

    button_fecha_parto = tk.Button(self, text="Fecha de Parto", command=self.calcular_fecha_parto, bg="#4CAF50", fg="white", width=15)

    button_fecha_parto.pack(pady=10)

    button_riesgo_embarazo = tk.Button(self, text="Riesgo de Embarazo", command=self.evaluar_riesgo_embarazo, bg="#f44336", fg="white", width=15)

    button_riesgo_embarazo.pack(pady=10)



  def calcular_fecha_parto(self):

    """Calcula la fecha de parto estimada."""



    ultima_menstruacion = self.entry_ultima_menstruacion.get()

    semanas_embarazo = int(self.entry_semanas_embarazo.get()) if self.entry_semanas_embarazo.get() else 0

    dias_embarazo = int(self.entry_dias_embarazo.get()) if self.entry_dias_embarazo.get() else 0



    try:

      fecha_parto = predecir_fecha_parto(ultima_menstruacion, semanas_embarazo, dias_embarazo)

      messagebox.showinfo("Fecha de Parto", f"Fecha de parto estimada: {fecha_parto}")

    except ValueError:

      messagebox.showerror("Error", "Ingresa una fecha válida (AAAA-MM-DD).")

# Función para evaluar el riesgo y mostrar los resultados
def evaluar_riesgo():
    edad = int(edad_entry.get())
    talla = int(talla_entry.get())
    peso = int(peso_entry.get())
    hipertension_cronica = hipertension_cronica_var.get()
    its = its_var.get()
    sobrepeso = sobrepeso_var.get()
    fumadora = fumadora_var.get()
    diabetes = diabetes_var.get()
    nuevos_datos = [[edad, talla, peso, hipertension_cronica, its, sobrepeso, fumadora, diabetes]]
    nuevos_datos = StandardScaler().fit_transform(nuevos_datos)
    riesgo_embarazo = model.predict(nuevos_datos)[0]

    respuesta_label.config(text=f"Riesgo de preeclampsia: {riesgo_embarazo}")

    consejo = obtener_consejo(riesgo_embarazo)
    consejo_label.config(text=consejo)

  def evaluar_riesgo_embarazo(self):


    # Carga de datos de prueba (reemplaza con tus datos reales)

    data = pd.DataFrame({

        'hipertension_arterial': [hipertension_arterial],

        'diabetes': [diabetes],

        'preeclampsia': [preeclampsia],

        'hipertension_cronica': [hipertension_cronica],

        'sobrepeso': [sobrepeso],

        'riesgo_embarazo': [0]  # Inicializa 'riesgo_embarazo' a 0

    })



    # Entrena la red neuronal

    model, scaler = entrenar_red_neuronal(data)



    # Predecir el riesgo de embarazo

    nuevos_datos = [[hipertension_arterial, diabetes, preeclampsia, hipertension_cronica, sobrepeso]]

    nuevos_datos = scaler.transform(nuevos_datos)

    prediccion = model.predict(nuevos_datos)[0][0]



    # Mostrar el resultado

    if prediccion >= 0.5:

      messagebox.showinfo("Riesgo de Embarazo", "El embarazo tiene un riesgo elevado.")

    else:

      messagebox.showinfo("Riesgo de Embarazo", "El embarazo no presenta un riesgo elevado.")



# Iniciar la aplicación

app = Aplicacion()

app.mainloop()
