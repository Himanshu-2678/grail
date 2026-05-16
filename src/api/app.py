import os
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Dict, Any, List
from src.engine.orchestrator import GrailOrchestrator

app = FastAPI(title="Grail AI - Constrained Conversational Retrieval")

# Mount static files
static_dir = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/")
async def root():
    return FileResponse(os.path.join(static_dir, "index.html"))

# Load orchestrator at startup
data_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'processed', 'normalized_catalog.json')
orchestrator = GrailOrchestrator(data_path)

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]

class Recommendation(BaseModel):
    name: str
    url: str
    test_type: str

class ChatResponse(BaseModel):
    reply: str
    recommendations: List[Recommendation]
    end_of_conversation: bool

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        # Pass messages directly to orchestrator to remain completely stateless
        response_data = orchestrator.process_chat([m.model_dump() for m in request.messages])
        
        return ChatResponse(
            reply=response_data["reply"],
            recommendations=response_data["recommendations"],
            end_of_conversation=response_data["end_of_conversation"]
        )
    except Exception as e:
        # Graceful fallback on critical crash
        print(f"Critical API Error: {e}")
        return ChatResponse(
            reply="I'm temporarily unable to process requests. Please try again.",
            recommendations=[],
            end_of_conversation=False
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.app:app", host="0.0.0.0", port=8000, reload=True)
