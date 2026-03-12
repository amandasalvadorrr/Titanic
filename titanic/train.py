import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# carregar dataset
df = pd.read_csv("titanic.csv")

# selecionar colunas importantes
df = df[["Survived","Pclass","Sex","Age","SibSp","Fare"]]

# transformar sexo em número
df["Sex"] = df["Sex"].map({
    "male":0,
    "female":1
})

# remover valores nulos
df = df.dropna()

# separar dados
X = df.drop("Survived", axis=1)
y = df["Survived"]

# dividir treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X,y,test_size=0.2,random_state=42
)

# criar modelo
model = RandomForestClassifier()

# treinar
model.fit(X_train,y_train)

# salvar modelo
joblib.dump(model,"modelo.pkl")

print("Modelo treinado e salvo!")