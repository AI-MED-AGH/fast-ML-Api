"""
Minimal containerized server using a typed request model.

Run locally:
    python examples/containerized/ml_server.py
"""

from typing import List

from pydantic import BaseModel, Field

from fastmlapi import MLController, postprocessing, preprocessing


class SumRequest(BaseModel):
    values: List[float] = Field(..., min_length=1, description="Numbers to score")


class ExampleModel(MLController):
    model_name = "example-model"
    model_version = "1.0.0"
    request_model = SumRequest
    request_example = {"values": [1.0, 2.0, 3.0]}

    def load_model(self):
        # Replace with real loading. Use self.cache_dir for downloaded weights,
        # e.g. os.environ.setdefault("HF_HOME", str(self.cache_dir / "hf")).
        return lambda values: sum(values)

    @preprocessing
    def preprocess(self, data: dict) -> List[float]:
        return data["values"]

    @postprocessing
    def postprocess(self, prediction: float) -> dict:
        return {"score": prediction}


# Importable as `ml_server:app` for `uvicorn`.
app = ExampleModel().app

if __name__ == "__main__":
    # Host/port come from FASTMLAPI_HOST / FASTMLAPI_PORT (default 0.0.0.0:8000).
    ExampleModel().run()
