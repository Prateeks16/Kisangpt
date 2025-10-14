# main.py
from fastapi import FastAPI, Request
from pydantic import BaseModel
from kisangpt_v3 import kisangpt_answer

app = FastAPI(title="KisanGPT API")

class Query(BaseModel):
    question: str
    state: str
    crop: str | None = None
    category: str | None = None
    sector: str | None = None
    npk: list[float] | None = None

@app.post("/kisangpt")
async def get_kisan_answer(req: Query):
    try:
        ans = kisangpt_answer(
            question=req.question,
            state=req.state,
            crop=req.crop,
            category=req.category,
            sector=req.sector,
            npk_values=req.npk,
        )
        return {"answer": ans}
    except Exception as e:
        return {"error": str(e)}

@app.get("/")
def root():
    return {"message": "🌾 KisanGPT API is running!"}
