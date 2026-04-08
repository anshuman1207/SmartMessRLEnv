from pydantic import BaseModel, ConfigDict
from typing import Dict, Any

class SmartMessAction(BaseModel):
    meals_to_prepare: float

class SmartMessReward(BaseModel):
    value: float
    metadata: Dict[str, Any] = {}

class SmartMessObservation(BaseModel):
    day_of_week: int        # 0=Mon, 6=Sun
    meal_type: int          # 0=Breakfast, 1=Lunch, 2=Dinner
    menu_type: int          # 0=Basic, 1=Special, 2=Biryani
    weather: int            # 0=Sunny, 1=Rainy/Stormy, 2=Extreme Heat
    special_event: int      # 0=Normal, 1=Fest, 2=Exam, 3=Holiday
    
    reward: SmartMessReward = SmartMessReward(value=0.0)
    done: bool = False
    metadata: Dict[str, Any] = {}
    
    model_config = ConfigDict(extra="allow")

class SmartMessState(BaseModel):
    step_count: int
    total_waste: float
    total_shortage: float
