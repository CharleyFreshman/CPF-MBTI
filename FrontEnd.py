from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import uvicorn

from BackEnd import MBTIPredictor

app = FastAPI(title="MBTI CPF 评估API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

predictor = None


@app.on_event("startup")
async def startup_event():
    global predictor
    predictor = MBTIPredictor('models/')


BOOKS_ENCODE = [0, 0.5, 1, 2, 4]


class CPFRequest(BaseModel):
    gender: int = Field(..., ge=0, le=1, description="0=女, 1=男")
    age: int = Field(..., ge=18, le=35)
    city_tier: int = Field(..., ge=1, le=5, description="18岁前常住地经济 1-5")
    area_type: int = Field(..., ge=1, le=3, description="1=农村,2=县城/乡镇,3=市区")
    material: int = Field(..., ge=1, le=5, description="物质生活条件")
    books: int = Field(..., ge=0, le=4, description="藏书量档位 0-4")
    guardian_edu: int = Field(..., ge=1, le=7)
    life_skill_req: int = Field(..., ge=1, le=5)
    culture_req: int = Field(..., ge=1, le=5)
    single_parent: int = Field(..., ge=0, le=1, description="0=双亲, 1=单亲")
    guardian_a_ie: int = Field(..., ge=0, le=1, description="监护人A 0=内向,1=外向")
    guardian_a_ft: int = Field(..., ge=0, le=1, description="监护人A 0=感性,1=理性")
    guardian_b_ie: int = Field(0, ge=0, le=1, description="监护人B 0=内向,1=外向）")
    guardian_b_ft: int = Field(0, ge=0, le=1, description="监护人B 0=感性,1=理性")
    life_satisfaction: int = Field(..., ge=1, le=5)
    life_meaning: int = Field(..., ge=1, le=5)
    education: int = Field(..., ge=1, le=7)
    peer_recognition: int = Field(..., ge=1, le=5)
    family_satisfaction: int = Field(..., ge=1, le=5)
    friendship_satisfaction: int = Field(..., ge=1, le=5)
    love_satisfaction: int = Field(..., ge=1, le=5)


class PredictionResponse(BaseModel):
    status: str
    confidence: float
    mbti_type: Optional[str] = None
    model: Optional[str] = None
    predictions: dict
    detailed: Optional[dict] = None


class RawRequest(BaseModel):
    data: List[float]
    single_parent: bool = False


def build_features(r: CPFRequest) -> List[float]:
    # 公共前缀: 性别,年龄,常住地经济,物质条件,藏书总量,监护人学历,文化知识,生活技能
    common = [
        r.gender, r.age, r.city_tier * r.area_type, r.material,
        BOOKS_ENCODE[r.books], r.guardian_edu, r.culture_req, r.life_skill_req,
    ]
    tail = [r.life_satisfaction, r.life_meaning, r.education, r.peer_recognition,
            r.family_satisfaction, r.friendship_satisfaction, r.love_satisfaction]
    if r.single_parent == 1:
        # 单亲模型(18特征): 是否单亲 + 监A
        return common + [1, r.guardian_a_ie, r.guardian_a_ft] + tail
    # 双亲模型(19特征): 监A + 监B (无 是否单亲)
    return common + [r.guardian_a_ie, r.guardian_a_ft, r.guardian_b_ie, r.guardian_b_ft] + tail


@app.get("/")
async def root():
    return {"message": "MBTI CPF 评估API", "version": "1.0"}


@app.get("/health")
async def health_check():
    return {"status": "healthy", "models_loaded": predictor is not None}


@app.post("/predict", response_model=PredictionResponse)
async def predict(req: CPFRequest):
    try:
        features = build_features(req)
        return predictor.get_formatted_result(features, single_parent=(req.single_parent == 1))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict_raw")
async def predict_raw(req: RawRequest):
    ms = predictor.single_parent if req.single_parent else predictor.two_parent
    n = len(ms.feature_names)
    if len(req.data) != n:
        raise HTTPException(status_code=400, detail=f"需要 {n} 个特征值")
    return predictor.get_formatted_result(req.data, single_parent=req.single_parent)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
