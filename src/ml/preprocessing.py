import pandas as pd
import pickle
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Télécharger les ressources nltk nécessaires (si ce n'est pas déjà fait)
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialiser le lemmatizer et les stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Charger les données
df = pd.read_csv("../../data/raw/tweet.csv")

# Mapper les sentiments en valeurs numériques
sentiment_mapping = {'negative': 0, 'neutral': 1, 'positive': 2}
df['sentiment'] = df['sentiment'].map(sentiment_mapping)

# Vérifier les valeurs manquantes
print(df.isnull().sum())

# Optionnel : Supprimer les lignes avec des valeurs manquantes
df.dropna(subset=["text", "sentiment"], inplace=True)

# Séparer les features et labels
X = df['text']  # Utiliser la colonne 'text'
y = df['sentiment']

# Nettoyer les données texte (fonction clean_text)
# Exemple de nettoyage sans stopwords
def clean_text(text):
    text = text.lower()  # Convertir en minuscules
    text = re.sub(r'http\S+|www\S+', '', text)  # Supprimer les URL
    text = re.sub(r'@\w+', '', text)  # Supprimer les mentions @
    text = text.translate(str.maketrans('', '', string.punctuation))  # Supprimer la ponctuation
    tokens = word_tokenize(text)  # Tokenisation
    # Ne pas enlever les stopwords
    tokens = [lemmatizer.lemmatize(word) for word in tokens]  # Lemmatisation sans filtrer par stopwords
    return ' '.join(tokens)


# Appliquer le nettoyage du texte
X_cleaned = X.apply(clean_text)

# Vectorisation avec Tfidf
vectorizer = TfidfVectorizer()
X_vectorized = vectorizer.fit_transform(X_cleaned)

# Sauvegarder le vectorizer pour la prédiction
with open("../../models/vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

# Séparer les données en train et test
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

# Entraîner le modèle Logistic Regression
model = LogisticRegression(max_iter=1000)
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
