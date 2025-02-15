# sentiment_analysis_project-
Sentiment Analysis Project  
📌 Description
Ce projet vise à analyser les sentiments de différents types de textes (tweets, avis Amazon, critiques de films, etc.) en utilisant trois approches distinctes :
Machine Learning (SVM, Naïve Bayes, Random Forest) 
Deep Learning (LSTM, BERT)  
Lexicons et Big Data (TextBlob, VADER, Spark NLP)

sentiment_analysis_project/
│── data/                   # Dossier des datasets (tweets, reviews Amazon, films)
│── models/                 # Modèles entraînés (fichiers .pkl ou .h5)
│── notebooks/              # Jupyter Notebooks pour l’exploration et tests
│── src/                    # Code source
│   │── ml/                 # Machine Learning (SVM, RF, NB…)
│   │── deep_learning/      # Deep Learning (LSTM, BERT…)
│   │── lexicon/            # Lexicons, VADER, TextBlob…
│   └── preprocessing.py    # Fonction de prétraitement commune
│── scripts/                # Scripts pour exécuter le projet
│   │── train.py            # Script d’entraînement des modèles
│   │── predict.py          # Script pour prédire sur un texte donné
│── results/                # Résultats et visualisations
│── requirements.txt        # Liste des bibliothèques nécessaires
│── README.md               # Documentation
└── .gitignore              # Exclure fichiers inutiles (datasets lourds, modèles…)
