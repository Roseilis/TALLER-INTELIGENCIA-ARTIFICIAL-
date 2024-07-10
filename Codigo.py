import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from tkinter import Tk, Label, Entry, Button, StringVar, IntVar, Radiobutton, messagebox
from tkinter import ttk
from datetime import datetime, timedelta




# Función para cargar datos históricos
def cargar_datos_historicos():
    try:
        data = pd.read_csv("datos_historicos_embarazo.csv")
        return data
    except FileNotFoundError:
        print("Archivo de datos históricos no encontrado.")
        return None

# Función para preprocesar los datos
def preprocesar_datos(data):
    X = data.drop("desarrollo_de_preeclampsia", axis=1)
    y = data["desarrollo_de_preeclampsia"]
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

# Función para entrenar el modelo
def entrenar_modelo(X_train, y_train):
    model = MLPClassifier(hidden_layer_sizes=(100, 50), activation="relu", solver="adam", max_iter=500)
    model.fit(X_train, y_train)
    return model

# Función para evaluar el modelo
def evaluar_modelo(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Precisión del modelo: {accuracy:.2f}")

# Función para obtener consejo
def obtener_consejo(riesgo):
    consejos = {
        0: "Su riesgo de desarrollar preeclampsia es bajo. ¡Disfrute de este tiempo especial!",
        1: "Su riesgo de desarrollar preeclampsia es alto. Consulte a su médico regularmente para un mejor seguimiento."
    }
    return consejos[riesgo]

# Función para calcular la fecha de parto
def calcular_fecha_parto(fecha_ultima_menstruacion):
    try:
        fecha_ultima_menstruacion = datetime.strptime(fecha_ultima_menstruacion, "%Y-%m-%d")
        fecha_parto = fecha_ultima_menstruacion + timedelta(days=280)
        return fecha_parto.strftime("%Y-%m-%d")
    except ValueError:
        messagebox.showerror("Error", "Formato de fecha inválido. Use YYYY-MM-DD")
        return None

# Función para evaluar el riesgo
def evaluar_riesgo():
    edad = int(edad_entry.get())
    talla = float(talla_entry.get())
    peso = float(peso_entry.get())
    hipertension_cronica = int(hipertension_cronica_var.get())
    its = int(its_var.get())
    sobrepeso = int(sobrepeso_var.get())
    fumadora = int(fumadora_var.get())
    diabetes = int(diabetes_var.get())

    nuevos_datos = [[edad, talla, peso, hipertension_cronica, its, sobrepeso, fumadora, diabetes]]
    nuevos_datos = StandardScaler().fit_transform(nuevos_datos)
    riesgo_embarazo = model.predict(nuevos_datos)[0]

    respuesta_label.config(text=f"Riesgo de preeclampsia: {riesgo_embarazo}")

    consejo = obtener_consejo(riesgo_embarazo)
    consejo_label.config(text=consejo)

  
# Función para calcular la fecha de parto
def calcular_fecha_parto():
    fecha_ultima_menstruacion = fecha_ultima_menstruacion_entry.get()
    fecha_parto = calcular_fecha_parto(fecha_ultima_menstruacion)
    if fecha_parto:
        fecha_parto_label.config(text=f"Fecha posible de parto: {fecha_parto}")

# Crear la ventana principal
window = Tk()
window.title("Evaluación de Riesgo de Embarazo")

# Imagen de fondo
background_image = "imagen_fondo.png"  # Reemplaza con el nombre de tu imagen
background_label = Label(window, image=background_image)
background_label.place(x=0, y=0, relwidth=1, relheight=1)

# Etiquetas y campos de entrada
edad_label = Label(window, text="Edad:")
edad_label.place(x=50, y=50)
edad_entry = Entry(window)
edad_entry.place(x=100, y=50)

talla_label = Label(window, text="Talla (cm):")
talla_label.place(x=50, y=80)
talla_entry = Entry(window)
talla_entry.place(x=100, y=80)

peso_label = Label(window, text="Peso (kg):")
peso_label.place(x=50, y=110)
peso_entry = Entry(window)
peso_entry.place(x=100, y=110)

# Variables para los botones de selección
hipertension_cronica_var = IntVar()
its_var = IntVar()
sobrepeso_var = IntVar()
fumadora_var = IntVar()
diabetes_var = IntVar()

# Botones de selección
hipertension_cronica_label = Label(window, text="Hipertensión crónica:")
hipertension_cronica_label.place(x=50, y=140)
hipertension_cronica_si = Radiobutton(window, text="Sí", variable=hipertension_cronica_var, value=1)
hipertension_cronica_si.place(x=150, y=140)
hipertension_cronica_no = Radiobutton(window, text="No", variable=hipertension_cronica_var, value=0)
hipertension_cronica_no.place(x=200, y=140)

its_label = Label(window, text="ITS:")
its_label.place(x=50, y=170)
its_si = Radiobutton(window, text="Sí", variable=its_var, value=1)
its_si.place(x=150, y=170)
its_no = Radiobutton(window, text="No", variable=its_var, value=0)
its_no.place(x=200, y=170)

sobrepeso_label = Label(window, text="Sobrepeso:")
sobrepeso_label.place(x=50, y=200)
sobrepeso_si = Radiobutton(window, text="Sí", variable=sobrepeso_var, value=1)
sobrepeso_si.place(x=150, y=200)
sobrepeso_no = Radiobutton(window, text="No", variable=sobrepeso_var, value=0)
sobrepeso_no.place(x=200, y=200)

fumadora_label = Label(window, text="Fumadora:")
fumadora_label.place(x=50, y=230)
fumadora_si = Radiobutton(window, text="Sí", variable=fumadora_var, value=1)
fumadora_si.place(x=150, y=230)
fumadora_no = Radiobutton(window, text="No", variable=fumadora_var, value=0)
fumadora_no.place(x=200, y=230)

diabetes_label = Label(window, text="Diabetes:")
diabetes_label.place(x=50, y=260)
diabetes_si = Radiobutton(window, text="Sí", variable=diabetes_var, value=1)
diabetes_si.place(x=150, y=260)
diabetes_no = Radiobutton(window, text="No", variable=diabetes_var, value=0)
diabetes_no.place(x=200, y=260)

# Botón para evaluar el riesgo
evaluar_riesgo_button = Button(window, text="Evaluar Riesgo", command=evaluar_riesgo)
evaluar_riesgo_button.place(x=100, y=300)

# Etiquetas para mostrar la respuesta
respuesta_label = Label(window, text="")
respuesta_label.place(x=100, y=350)

consejo_label = Label(window, text="")
consejo_label.place(x=100, y=380)

# Campo de entrada y botón para la fecha de última menstruación
fecha_ultima_menstruacion_label = Label(window, text="Fecha de última menstruación (YYYY-MM-DD):")
fecha_ultima_menstruacion_label.place(x=50, y=450)
fecha_ultima_menstruacion_entry = Entry(window)
fecha_ultima_menstruacion_entry.place(x=100, y=450)
calcular_fecha_parto_button = Button(window, text="Calcular Fecha de Parto", command=calcular_fecha_parto)
calcular_fecha_parto_button.place(x=100, y=480)

# Etiqueta para mostrar la fecha de parto
fecha_parto_label = Label(window, text="")
fecha_parto_label.place(x=100, y=510)

# Cargar y entrenar el modelo
data = cargar_datos_historicos()
if data is not None:
    X_train, X_test, y_train, y_test = preprocesar_datos(data)
    model = entrenar_modelo(X_train, y_train)
    evaluar_modelo(model, X_test, y_test)
else:
    messagebox.showerror("Error", "No se pudo cargar el archivo de datos históricos.")

# Iniciar la ventana
window.mainloop()
