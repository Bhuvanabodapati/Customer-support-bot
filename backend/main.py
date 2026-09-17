from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel

from backend.chatbot import get_response

app = FastAPI(
    title="AI Customer Support Bot",
    description="ML-powered customer support chatbot API",
    version="1.0"
)


class ChatRequest(BaseModel):
    message: str


# Serve frontend
@app.get("/")
def home():
    return FileResponse("frontend/index.html")


# Chat endpoint
@app.post("/chat")
def chat(request: ChatRequest):

    intent, response, confidence = get_response(request.message)

    return {
    "intent": intent,
    "response": response,
    "confidence": confidence
}