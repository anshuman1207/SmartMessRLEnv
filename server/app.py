import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, Any, Optional
from environment import SmartMessEnvironment
from models import SmartMessAction

app = FastAPI(title="SmartMess OpenEnv")

# Use a global dictionary or single instance.
# For evaluation, a global instance is simplest and meets HF Space stateless evaluation requirements.
global_env = SmartMessEnvironment(task_level="hard")

class ResetRequest(BaseModel):
    task_level: Optional[str] = "hard"

@app.post("/reset")
def reset(req: ResetRequest = None):
    level = req.task_level if req else "hard"
    global global_env
    global_env = SmartMessEnvironment(task_level=level)
    obs = global_env.reset()
    
    return {
        "observation": obs.model_dump(),
        "reward": obs.reward.model_dump(),
        "done": obs.done,
        "info": {}
    }

@app.post("/step")
def step(action: SmartMessAction):
    obs = global_env.step(action)
    return {
        "observation": obs.model_dump(),
        "reward": obs.reward.model_dump(),
        "done": obs.done,
        "info": obs.metadata
    }

@app.get("/state")
def state():
    st = global_env.state()
    return st.model_dump()

def main():
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
