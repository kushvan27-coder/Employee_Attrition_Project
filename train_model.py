import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv("employee_attrition.csv")
X = df.drop("Attrition", axis=1)
y = df["Attrition"]

categorical = ["Department", "JobRole"]
numeric = [c for c in X.columns if c not in categorical]

preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
    ("num", StandardScaler(), numeric)
])

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=250, random_state=42, class_weight="balanced"
    ))
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

model.fit(X_train, y_train)
pred = model.predict(X_test)

print("Model Accuracy:", round(accuracy_score(y_test, pred)*100, 2), "%")
print(classification_report(y_test, pred))

joblib.dump(model, "attrition_model.pkl")
print("Model saved as attrition_model.pkl")
