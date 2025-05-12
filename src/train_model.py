import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report

from src.preprocessing import DataPreprocessor
from src.vectorization import Vectorizer

def train_model():
    data = pd.read_csv("data/data_one_cat.csv")

    preprocessor = DataPreprocessor()
    data["processed_text"] = data["Text"].apply(preprocessor.preprocess)
    
    vectorizer = Vectorizer(method="sbert")

    X = data["processed_text"]
    y = data["IsToxic"]

    X_train_text, X_test_text, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.fit_transform(X_test_text)

    # clf = LogisticRegression()
    clf = SVC()
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))

train_model()