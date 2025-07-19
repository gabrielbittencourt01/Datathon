import pandas as pd
import numpy as np
import joblib
import os
import re
import nltk

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, f1_score
from sklearn.utils import shuffle

from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer

# ⬇ Baixar recursos do nltk
nltk.download('stopwords')
nltk.download('rslp')
stop_words = set(stopwords.words('portuguese'))
stemmer = RSLPStemmer()

# 🔧 Função de pré-processamento
def preprocess(text):
    text = str(text).lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    tokens = text.split()
    tokens = [stemmer.stem(t) for t in tokens if t not in stop_words]
    return " ".join(tokens)

# 📦 Carregar dados
X_train = joblib.load("models/X_train.pkl")
y_train = joblib.load("models/y_train.pkl")
X_test = joblib.load("models/X_test.pkl")
y_test = joblib.load("models/y_test.pkl")

# 🎯 Selecionar texto bruto
X_train_text = [preprocess(t) for t in X_train[:, 0]]
X_test_text = [preprocess(t) for t in X_test[:, 0]]

# 🔡 Vetorização
vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1, 2))
X_train_vec = vectorizer.fit_transform(X_train_text)
X_test_vec = vectorizer.transform(X_test_text)

# ✅ Balanceamento manual (oversampling da classe 1)
X_df = pd.DataFrame(X_train_vec.todense())
X_df["target"] = y_train.values
df_min = X_df[X_df["target"] == 1]
df_maj = X_df[X_df["target"] == 0]
df_min_upsampled = df_min.sample(n=len(df_maj), replace=True, random_state=42)
df_bal = pd.concat([df_maj, df_min_upsampled])
y_bal = df_bal["target"].values
X_bal = df_bal.drop(columns=["target"]).values

# 🤖 Modelos
models = {
    "LogisticRegression": LogisticRegression(max_iter=1000, class_weight='balanced'),
    "RandomForest": RandomForestClassifier(n_estimators=150, class_weight='balanced'),
    "XGBoost": XGBClassifier(n_estimators=150, use_label_encoder=False, eval_metric='logloss'),
    "VotingClassifier": VotingClassifier(estimators=[
        ('lr', LogisticRegression(max_iter=1000, class_weight='balanced')),
        ('rf', RandomForestClassifier(n_estimators=100, class_weight='balanced')),
        ('xgb', XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='logloss'))
    ], voting='soft')
}

# 🔁 Validação cruzada
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
melhor_f1 = 0
melhor_modelo = None

for nome, modelo in models.items():
    print(f"\n📊 Avaliando {nome}...")
    f1_scores = cross_val_score(modelo, X_bal, y_bal, scoring='f1', cv=cv, n_jobs=-1)
    print(f"F1-score médio (validação cruzada): {f1_scores.mean():.4f}")

    modelo.fit(X_bal, y_bal)
    y_pred = modelo.predict(X_test_vec)
    f1 = f1_score(y_test, y_pred)
    print("Relatório no conjunto de teste:")
    print(classification_report(y_test, y_pred))

    if f1 > melhor_f1:
        melhor_f1 = f1
        melhor_modelo = modelo
        joblib.dump(melhor_modelo, "models/melhor_modelo.pkl")
        joblib.dump(vectorizer, "models/melhor_vectorizer.pkl")

print(f"\n✅ Melhor modelo salvo: {melhor_modelo.__class__.__name__} (F1 = {melhor_f1:.4f})")
