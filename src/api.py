from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel

from src.inference import AlzheimerChatbot


app = FastAPI(
    title="Alzheimer's Medical Chatbot API",
    description="API for Alzheimer's caregiver and patient support chatbot.",
    version="1.0.0",
)

chatbot: Optional[AlzheimerChatbot] = None


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    response: str


@app.on_event("startup")
def startup_event():
    global chatbot
    chatbot = AlzheimerChatbot()


@app.get("/")
def root():
    return {
        "message": "Alzheimer's Chatbot API is running.",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": chatbot is not None,
    }


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    global chatbot

    if chatbot is None:
        chatbot = AlzheimerChatbot()

    answer = chatbot.chat(request.message)

    return ChatResponse(response=answer)