# Copyright 2025 Pouria Sarmasti

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.config_reader import read_config

config_path = "config/config.yaml"
web_config = read_config(config_path)["web"]

model_dir = web_config["model_output_dir"]
model_name = web_config["model_name"]


class DemanddRequest(BaseModel):
    hour_of_day: int = Field(ge=0, le=23)
    day: int = Field(ge=0)
    row: int = Field(ge=0, le=7)
    col: int = Field(ge=0, le=7)


class DemandResponse(BaseModel):
    demand: int


ml_models = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model
    ml_models[Path(model_name).stem] = joblib.load(Path(model_dir) / model_name)
    yield
    # Clean up the ML models and release the resources
    ml_models.clear()


app = FastAPI(lifespan=lifespan)


@app.post("/predictin", response_model=DemandResponse)
async def predict_demand(request: DemanddRequest):

    features = pd.DataFrame(
        [
            {
                "hour_of_day": request.hour_of_day,
                "day": request.day,
                "row": request.row,
                "col": request.col,
            }
        ]
    )

    prediction = ml_models[Path(model_name).stem].predict(features)[0]

    return {"demand": round(prediction)}


if __name__ == "__main__":
    uvicorn.run(
        "web.application:app",
        host=os.getenv("WEB_HOST", web_config["host"]),
        port=int(os.getenv("WEB_PORT", web_config["port"])),
        reload=True,
    )
