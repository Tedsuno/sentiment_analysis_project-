import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier  # ou LogisticRegression, si tu préfères un modèle plus rapide
from sklearn.metrics import classification_report

# Charger les données
df = pd.read_csv("../../data/raw/tweet.csv")

# Gérer les NaN
df['text'] = df['text'].fillna('')  # Remplacer les NaN par une chaîne vide

# Séparer les features et labels
X = df["text"]
y = df["sentiment"]

# Vectoriser le texte avec une taille de vocabulaire plus petite
vectorizer = TfidfVectorizer(max_features=10000)  # Limiter à 10 000 mots
X_vectorized = vectorizer.fit_transform(X)

# Sauvegarder le vectorizer
with open("../../models/vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

# Séparer en train et test
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

# Entraîner un modèle plus rapide
model = RandomForestClassifier(n_estimators=20, class_weight='balanced', random_state=42, n_jobs=-1)  # Utilisation du parallélisme
model.fit(X_train, y_train)

# Sauvegarder le modèle entraîné
with open("../../models/ml_model.pkl", "wb") as f:
    pickle.dump(model, f)

# Évaluer le modèle
y_pred = model.predict(X_test)
report = classification_report(y_test, y_pred)

# Sauvegarder le rapport de classification
with open("../../results/classification_report.txt", "w") as f:
    f.write(report)

print("✅ Modèle ML entraîné et sauvegardé.")
