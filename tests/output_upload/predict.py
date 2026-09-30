from cog import BasePredictor, BaseModel, Path
from typing import Optional
class Output(BaseModel):
    video: Path
    metadata: Path
    seed: int
    frames: Optional[Path] = None
class Predictor(BasePredictor):
    def predict(self) -> Output:
        v=Path('/tmp/test.mp4'); v.write_bytes(b'test-mp4-placeholder')
        m=Path('/tmp/test.json'); m.write_text('{"seed":42}')
        return Output(video=v,metadata=m,seed=42)
