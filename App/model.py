# Model Libraries
import joblib
import pickle
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

###### Preprocessing functions ######
def train_model():
    file_path = "roles-based-on-skills.csv"
    df = pd.read_csv(file_path)

    # Data Cleaning
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    df["ALL"] = df["ALL"].str.lower().apply(lambda x: re.sub(r'[^a-zA-Z0-9, ]', '', x).strip()) #removing special characters
    df["Target"] = df["Target"].str.lower().str.strip() #Trim the target column (removing spaces)

    # Extract features and target variable
    X = df["ALL"]
    y = df["Target"]

    # Convert text data into TF-IDF features
    vectorizer = TfidfVectorizer(stop_words="english", max_features=10000)
    X_tfidf = vectorizer.fit_transform(X)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

    # Define individual models
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    svm_model = SVC(kernel='linear', probability=True, random_state=42)
    log_reg = LogisticRegression(max_iter=1000, random_state=42)
    knn_model = KNeighborsClassifier(n_neighbors=5)

    # Combine models using VotingClassifier
    ensemble_model = VotingClassifier(estimators=[
        ('rf', rf_model),
        ('svm', svm_model),
        ('lr', log_reg),
        ('knn', knn_model)
    ], voting='soft')

    # Train the ensemble model
    ensemble_model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = ensemble_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Ensemble Model Accuracy: {accuracy:.2f}")

    # Save the trained model and vectorizer
    joblib.dump(ensemble_model, "ensemble_model.pkl")
    pickle.dump(vectorizer, open("tfidf_vectorizer.pkl", "wb"))


train_model()