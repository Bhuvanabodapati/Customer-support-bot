import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


# Get project root directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# Load dataset
data_path = os.path.join(BASE_DIR, "data", "faq.csv")
data = pd.read_csv(data_path)


# Input and output
X = data["question"]
y = data["intent"]


# Create ML pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2)
    )),
    ("classifier", LogisticRegression(
        max_iter=1000
    ))
])


# Train model
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# Train model on training data
model.fit(X_train, y_train)

print("Model trained successfully!")


# Evaluate model on unseen test data
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\n--- Model Evaluation ---")
print(f"Accuracy: {accuracy:.2%}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# Retrain final model using the complete dataset
model.fit(X, y)

print(f"\nFinal model trained using all {len(data)} examples.")
print(f"Training examples: {len(data)}")
print(f"Number of intents: {y.nunique()}")
print(f"Intents: {sorted(y.unique())}")


# Create model folder if it doesn't exist
model_dir = os.path.join(BASE_DIR, "model")
os.makedirs(model_dir, exist_ok=True)


# Save trained model
model_path = os.path.join(
    model_dir,
    "customer_support_model.pkl"
)

joblib.dump(model, model_path)

print("Model saved successfully!")
print(f"Saved to: {model_path}")


# Test questions
test_questions = [
    "Where is my package?",
    "Can I track my order?",
    "When will my order arrive?",
    "I need my money back",
    "My payment failed",
    "I forgot my password",
    "I want to cancel my order",
    "I want to talk to an agent"
]


# Predictions
predictions = model.predict(test_questions)

print("\n--- Test Predictions ---")

for question, prediction in zip(test_questions, predictions):
    print(f"\nQuestion: {question}")
    print(f"Predicted Intent: {prediction}")



# --------------------------------------------------
# Evaluate on independent test dataset
# --------------------------------------------------

test_data_path = os.path.join(
    BASE_DIR,
    "data",
    "test_faq.csv"
)

test_data = pd.read_csv(test_data_path)

X_external = test_data["question"]
y_external = test_data["intent"]


# Predict external test questions
external_predictions = model.predict(X_external)


# Calculate external accuracy
external_accuracy = accuracy_score(
    y_external,
    external_predictions
)

print("\n--- Independent Test Evaluation ---")
print(f"Test examples: {len(test_data)}")
print(f"Independent Test Accuracy: {external_accuracy:.2%}")


# Detailed evaluation
print("\nClassification Report:")
print(
    classification_report(
        y_external,
        external_predictions
    )
)


# Confusion matrix
print("\nConfusion Matrix:")
print(
    confusion_matrix(
        y_external,
        external_predictions
    )
)