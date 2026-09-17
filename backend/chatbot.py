import pandas as pd
import joblib
import os


# Get project root directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# Load trained model
model_path = os.path.join(
    BASE_DIR,
    "model",
    "customer_support_model.pkl"
)

model = joblib.load(model_path)


# Load FAQ dataset
data_path = os.path.join(
    BASE_DIR,
    "data",
    "faq.csv"
)

data = pd.read_csv(data_path)


def get_response(user_question):

    # Predict intent
    predicted_intent = model.predict([user_question])[0]

    # Calculate confidence
    probabilities = model.predict_proba([user_question])[0]
    confidence = max(probabilities)

    # Confidence threshold
    CONFIDENCE_THRESHOLD = 0.45

    # If confidence is too low, use fallback response
    if confidence < CONFIDENCE_THRESHOLD:
        response = (
            "Sorry, I couldn't understand your request. "
            "Could you please rephrase your question?"
        )

        return "unknown", response, confidence

    # Find matching responses
    matching_rows = data[data["intent"] == predicted_intent]

    if not matching_rows.empty:
        response = matching_rows.iloc[0]["answer"]
    else:
        response = "Sorry, I don't have an answer for that."

    return predicted_intent, response, confidence


# Test chatbot directly
if __name__ == "__main__":

    print("🤖 Customer Support Bot")
    print("Type 'exit' to stop the chatbot.\n")

    while True:

        user_question = input("You: ")

        if user_question.lower() == "exit":
            print("Bot: Thank you for contacting customer support!")
            break

        intent, response, confidence = get_response(user_question)

        print(f"Bot: {response}")
        print(f"(Detected Intent: {intent})")
        print(f"(Confidence: {confidence:.2%})\n")