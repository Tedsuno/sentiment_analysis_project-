import pandas as pd
import pickle
from preprocessing import clean_text  # La fonction de nettoyage doit être identique à celle utilisée lors de l'entraînement

# Charger les données
df = pd.read_csv("../../data/raw/tweet.csv")

# Remplacer les NaN dans la colonne 'text'
df['text'] = df['text'].fillna('')

# Charger le modèle et le vectorizer sauvegardés
with open('../../models/ml_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('../../models/vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Appliquer le prétraitement sur tous les tweets
df['cleaned_text'] = df['text'].apply(clean_text)

# Vectoriser les tweets nettoyés
X_vectorized = vectorizer.transform(df['cleaned_text'])

# Faire les prédictions
predictions = model.predict(X_vectorized)

# Ajouter une nouvelle colonne 'predicted_sentiment' dans le DataFrame
df['predicted_sentiment'] = predictions

# Comparer les prédictions aux sentiments réels
# Afficher quelques lignes pour vérifier
print(df[['text', 'sentiment', 'predicted_sentiment']].head())

# Sauvegarder les résultats dans un fichier CSV pour analyse
df.to_csv("../../results/tweet_predictions.csv", index=False)

# Afficher un message de confirmation
print("✅ Les prédictions ont été ajoutées à la colonne 'predicted_sentiment'. Le fichier a été sauvegardé.")
