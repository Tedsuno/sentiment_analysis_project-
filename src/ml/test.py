import pandas as pd
import pickle
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# Télécharger les ressources NLTK
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Charger les données
df = pd.read_csv("../../data/raw/tweet.csv")

# Supprimer les lignes avec des valeurs manquantes dans la colonne 'text'
df.dropna(subset=['text'], inplace=True)

# Mapper les sentiments en valeurs numériques
sentiment_mapping = {'negative': 0, 'neutral': 1, 'positive': 2}
df['sentiment'] = df['sentiment'].map(sentiment_mapping)

# Vérifier la répartition des classes
print("Répartition des classes :")
print(df['sentiment'].value_counts())

# Optionnel : Appliquer un sur-échantillonnage (SMOTE) pour équilibrer les classes
X = df['text']
y = df['sentiment']

# Nettoyage des données texte
def clean_text(text):
    # Vérifier si text est une chaîne de caractères, sinon retourner une chaîne vide
    if not isinstance(text, str):
        return ''
    
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    text = text.lower()  # Convertir en minuscules
    text = re.sub(r'http\S+|www\S+', '', text)  # Supprimer les URL
    text = re.sub(r'@\w+', '', text)  # Supprimer les mentions @
    text = text.translate(str.maketrans('', '', string.punctuation))  # Supprimer la ponctuation
    tokens = word_tokenize(text)  # Tokenisation
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]  # Lemmatisation et stopwords
    return ' '.join(tokens)

# Appliquer le nettoyage du texte
X_cleaned = X.apply(clean_text)

# Affichage d'un échantillon avant et après nettoyage
print("Exemple de texte avant et après nettoyage :")
print("Avant nettoyage : ", X.iloc[0])
print("Après nettoyage : ", X_cleaned.iloc[0])

# Vectorisation avec Tfidf
vectorizer = TfidfVectorizer()
X_vectorized = vectorizer.fit_transform(X_cleaned)

# Sauvegarder le vectorizer pour la prédiction
with open("../../models/vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

# Séparer les données en train et test
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

# Appliquer SMOTE pour équilibrer les classes dans les données d'entraînement
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# Entraîner le modèle Logistic Regression
model = LogisticRegression(max_iter=1000, C=0.1)
model.fit(X_train_res, y_train_res)

# Sauvegarder le modèle entraîné
with open("../../models/ml_model.pkl", "wb") as f:
    pickle.dump(model, f)

# Prédire avec le modèle
y_pred = model.predict(X_test)

# Afficher le rapport de classification
report = classification_report(y_test, y_pred)
print("Rapport de classification :")
print(report)

# Sauvegarder le rapport de classification
with open("../../results/classification_report.txt", "w") as f:
    f.write(report)

# Afficher la matrice de confusion
cm = confusion_matrix(y_test, y_pred)
print("Matrice de confusion :")
print(cm)

# Visualiser la matrice de confusion
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Neutral', 'Positive'], yticklabels=['Negative', 'Neutral', 'Positive'])
plt.xlabel('Prédiction')
plt.ylabel('Vrai')
plt.title('Matrice de Confusion')
plt.show()

print("✅ Modèle ML entraîné et évalué.")
