# sentiment_analysis_project-
Sentiment Analysis Project  
📌 Description
Ce projet vise à analyser les sentiments de différents types de textes (tweets, avis Amazon, critiques de films, etc.) en utilisant trois approches distinctes :
Machine Learning (SVM, Naïve Bayes, Random Forest) 
Deep Learning (LSTM, BERT)  
Lexicons et Big Data (TextBlob, VADER, Spark NLP)

sentiment_analysis_project/
│── data/                   # Dossier des datasets (tweets, reviews Amazon, films)\n
│── models/                 # Modèles entraînés (fichiers .pkl ou .h5)\n
│── notebooks/              # Jupyter Notebooks pour l’exploration et \n
│── src/                    # Code source\n
│   │── ml/                 # Machine Learning (SVM, RF, NB…)\n
│   │── deep_learning/      # Deep Learning (LSTM, BERT…)\n
│   │── lexicon/            # Lexicons, VADER, TextBlob…\n
│   └── preprocessing.py    # Fonction de prétraitement commune\n
│── scripts/                # Scripts pour exécuter le projet\n
│   │── train.py            # Script d’entraînement des modèles\n
│   │── predict.py          # Script pour prédire sur un texte donné\n
│── results/                # Résultats et visualisations\n
│── requirements.txt        # Liste des bibliothèques nécessaires\n
│── README.md               # Documentation\n
└── .gitignore              # Exclure fichiers inutiles (datasets lourds, modèles…)\n
