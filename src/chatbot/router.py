from fastapi import APIRouter, HTTPException
# Import the service chat under a different name to avoid shadowing
from .service import chat as chat_service, tool_calling as tool_calling_service
from .schema import ChatRequest, ChatResponse

router = APIRouter(prefix="/chatbot", tags=["chatbot"])


@router.post('/chat', response_model=ChatResponse)
async def chat(payload: ChatRequest):
    """General chat endpoint: forwards `payload.message` to the LLM service.

    Uses `service.rag` under the hood but without retrieved documents (empty list),
    providing a simple conversational interface.
    """
    try:
        # Call the imported service function (not the local handler)
        resp = chat_service(payload.message)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error running chatbot: {e}")

    answer = str(resp)
    return ChatResponse(answer=answer)

@router.post('/tool_calling', response_model=ChatResponse)
async def tool_calling(payload: ChatRequest):
    try:
        # Call the imported service function (not the local handler)
        resp = tool_calling_service(payload.message)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error running chatbot: {e}")

    answer = str(resp)
    return ChatResponse(answer=answer)