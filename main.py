from fastapi import FastAPI

from src.recognition.router import router as recognition_router
from src.chatbot.router import router as chatbot_router


def create_app() -> FastAPI:
    app = FastAPI(title="SnorlaxCode Cloud")
    app.include_router(recognition_router)
    app.include_router(chatbot_router)
    return app


app = create_app()


if __name__ == '__main__':
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
