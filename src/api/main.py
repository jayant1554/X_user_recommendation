from fastapi import FastAPI

from src.api.routes import router

app = FastAPI(
    title="User Recommendation API",
    version="1.0.0",
)

app.include_router(router)


@app.get("/")
def root():
    return {"message": "User Recommendation API is running."}