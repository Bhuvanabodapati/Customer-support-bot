🤖 AI Customer Support Bot

NLP-Based Customer Query Classification & Automated Response System

An NLP-based AI Customer Support Bot that automatically classifies
customer queries into predefined support intents using TF-IDF and
Logistic Regression, and provides relevant responses through a
FastAPI backend and interactive web interface.

🚀 Live Demo

🔗 Live Demo: https://customer-support-bot-k5ma.onrender.com

Deployed using Render with a FastAPI backend.

💻 Source Code

🔗 GitHub Repository:
https://github.com/Bhuvanabodapati/Customer-support-bot

📌 Project Overview

Customer support systems receive a large number of repetitive queries
related to orders, payments, refunds, deliveries, account issues, and
cancellations.

The AI Customer Support Bot uses Natural Language Processing
(NLP) and Machine Learning to identify the intent behind a
customer's question and provide an appropriate automated response.

The system converts text into numerical features using TF-IDF,
predicts the customer's intent using a Logistic Regression
classifier, and maps the predicted intent to an FAQ response.

A confidence-based fallback mechanism is also implemented to handle
queries that the model cannot classify with sufficient confidence.

🔄 Workflow

Customer Query
      ↓
Text Input
      ↓
TF-IDF Vectorization
      ↓
Logistic Regression
      ↓
Intent Prediction
      ↓
Confidence Check
      ↓
FAQ Response / Fallback Response
      ↓
Web Interface

🎯 Problem Statement

Traditional customer support systems often require human agents to
handle repetitive queries such as:

Where is my order?

Can I cancel my order?

How can I get a refund?

Why did my payment fail?

When will my order be delivered?

How can I contact customer support?

How can I manage my account?

Handling a large number of similar queries manually can increase support
workload and response time.

Objective

The objective of this project is to develop a lightweight Machine
Learning-based customer support chatbot that can:

Understand common customer queries.

Identify the intent behind each query.

Classify queries into predefined support categories.

Provide automated responses.

Detect low-confidence or unsupported queries.

Integrate a Machine Learning model with a REST API.

Provide an interactive web-based chatbot interface.

✨ Key Features

🤖 1. Intent Classification

The chatbot classifies customer queries into 7 predefined support
intents:

Intent              Description

account           Account-related queries
cancellation      Order cancellation requests
contact_support   Customer support contact requests
delivery          Delivery and shipping queries
order_status      Order tracking and status queries
payment           Payment-related issues
refund            Refund-related queries

🧠 2. NLP-Based Text Processing

Customer questions are converted into numerical features using TF-IDF
(Term Frequency-Inverse Document Frequency).

The model uses unigrams and bigrams:

TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2)
)

This allows the model to learn individual words and short phrases such
as:

"track order"
"cancel order"
"payment failed"
"refund money"

📊 3. Machine Learning Classification

The project uses Logistic Regression for intent classification.

LogisticRegression(
    max_iter=1000
)

Logistic Regression was selected because it is efficient for text
classification, fast to train, suitable for small and medium-sized
datasets, simple to implement, and capable of generating class
probabilities for confidence-based prediction.

🎯 4. Confidence-Based Fallback

The chatbot uses a confidence threshold of:

0.45

If the model's confidence is below this threshold, the chatbot returns:

Sorry, I couldn't understand your request. Could you please rephrase
your question?

This helps prevent unrelated responses for unsupported queries.

⚡ 5. FastAPI Backend

The trained Machine Learning model is integrated with a FastAPI backend.

Main endpoint:

POST /chat

Example request:

{
  "message": "Where is my order?"
}

Example response:

{
  "intent": "order_status",
  "response": "You can check your order status using your order ID.",
  "confidence": 0.49
}

💻 6. Interactive Web Interface

The frontend is developed using:

HTML

CSS

JavaScript

Users can enter questions and receive responses from the Machine
Learning model in real time.

🧠 Machine Learning Approach

1. Dataset Preparation

A custom FAQ dataset was created containing customer questions,
responses, and their corresponding intents.

Dataset Property                Value

Training Examples             130
Supported Intents               7
Independent Test Examples      35
Test Examples per Intent        5

The dataset was improved iteratively by adding representative examples,
especially for closely related categories such as Payment, Refund, and
Order Status.

2. TF-IDF Vectorization

The first stage converts text into numerical features using TF-IDF.

TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2)
)

TF-IDF gives greater importance to words and phrases that help
distinguish between different intents.

3. Logistic Regression

After vectorization, the TF-IDF features are passed to a Logistic
Regression classifier.

LogisticRegression(
    max_iter=1000
)

4. Model Pipeline

FAQ Dataset
     ↓
Text Questions
     ↓
TF-IDF Vectorization
     ↓
Logistic Regression
     ↓
Intent Prediction
     ↓
Confidence Score
     ↓
FAQ Response

📊 Model Performance

The final model was evaluated using an independent test dataset
containing 35 unseen customer queries.

Final Model

Algorithm: Logistic Regression

Feature Extraction: TF-IDF

N-gram Range: (1, 2)

Confidence Threshold: 0.45

Overall Performance

Metric                             Score

Training Examples                130
Supported Intents                  7
Independent Test Examples         35
Correct Predictions               34
Incorrect Predictions              1
Accuracy                      97.14%
Macro Precision                 0.98
Macro Recall                    0.97
Macro F1-Score                  0.97

The independent test dataset contains 5 examples per intent. These
results represent performance on this evaluation dataset and should
not be interpreted as guaranteed real-world accuracy.

📈 Intent-Wise Performance

Intent               Precision    Recall    F1-Score   Support

Account                1.00        1.00       1.00        5
Cancellation           1.00        1.00       1.00        5
Contact Support        1.00        1.00       1.00        5
Delivery               1.00        1.00       1.00        5
Order Status           1.00        1.00       1.00        5
Payment                0.83        1.00       0.91        5
Refund                 1.00        0.80       0.89        5
Macro Average    0.98    0.97   0.97   35

🔲 Confusion Matrix

The confusion matrix below represents predictions made on the 35
independent test queries.

                  Predicted
              Account Cancel Contact Delivery Order Payment Refund

Account          5       0      0       0       0      0      0
Cancellation     0       5      0       0       0      0      0
Contact Support  0       0      5       0       0      0      0
Delivery         0       0      0       5       0      0      0
Order Status     0       0      0       0       5      0      0
Payment          0       0      0       0       0      5      0
Refund           0       0      0       0       0      1      4

Error Analysis

The model correctly classified 34 / 35 independent test queries.

The only misclassification was:

Actual:    Refund
Predicted: Payment

The error occurred between the refund and payment categories,
which contain some semantically related terminology.

🛠️ Technologies Used

Python

Pandas

Scikit-learn

Joblib

FastAPI

Uvicorn

HTML

CSS

JavaScript

Render

📁 Project Structure

Customer-support-bot/
│
├── backend/
│   ├── chatbot.py
│   └── main.py
│
├── data/
│   ├── faq.csv
│   └── test_faq.csv
│
├── frontend/
│   └── index.html
│
├── model/
│   ├── train_model.py
│   └── customer_support_model.pkl
│
├── screenshots/
│   ├── chatbot-home.png
│   ├── chatbot-order-status.png
│   ├── chatbot-refund.png
│   └── chatbot-unknown-query.png
│
├── tests/
├── .gitignore
├── README.md
└── requirements.txt

⚙️ Installation

Clone the repository:

git clone https://github.com/Bhuvanabodapati/Customer-support-bot.git
cd Customer-support-bot

Install dependencies:

pip install -r requirements.txt

▶️ Run Locally

Start the FastAPI application:

uvicorn backend.main:app --reload

Open:

http://127.0.0.1:8000

FastAPI documentation:

http://127.0.0.1:8000/docs

🌐 Deployment

The application is deployed using Render.

Build Command:
pip install -r requirements.txt

Start Command:
uvicorn backend.main:app --host 0.0.0.0 --port $PORT

📸 Application Preview

Chatbot Interface



Order Status Query



Refund Query



Unknown Query Handling



🔮 Future Enhancements

Add more customer support intents.

Expand the training dataset with more natural-language variations.

Add conversation history.

Add authentication and user sessions.

Integrate a database for real order-status lookup.

Add multilingual customer support.

Add a human-agent escalation workflow.

Improve intent classification with advanced NLP models.

👩‍💻 Author

Bhuvana Bodapati

GitHub: https://github.com/Bhuvanabodapati