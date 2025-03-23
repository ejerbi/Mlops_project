import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import pickle
import mlflow
import mlflow.sklearn

# Charger les données
file_path = "Loan_Data.csv"
df = pd.read_csv(file_path)

# Initialiser MLflow
mlflow.set_experiment("Suivi des modèles")
# Afficher les premières lignes et les informations générales



# Suppression de l'ID client
df_cleaned = df.drop(columns=["customer_id"])

# Séparer les features et la cible
X = df_cleaned.drop(columns=["default"])
y = df_cleaned["default"]

# Division en train et test (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Normalisation des variables continues
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Vérification des dimensions après prétraitement
X_train_scaled.shape, X_test_scaled.shape

with mlflow.start_run():
    # Entraîner le modèle
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Prédictions et calcul des métriques
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Enregistrer les métriques
    mlflow.log_metric("accuracy", accuracy)

    # Enregistrer le modèle
    mlflow.sklearn.log_model(model, "random_forest_model")

    print(f"Accuracy enregistrée dans MLflow: {accuracy}")



# Prédire les valeurs sur l'ensemble de test
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)

print("Modèle sauvegardé sous 'model.pkl' !")

