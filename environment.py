import random
import json
import os
from models import SmartMessObservation, SmartMessAction, SmartMessState, SmartMessReward
from forecast_core import predict_demand
from optimizer_core import optimize_cooking

WEATHER_MAP = {0: "Sunny", 1: "Rainy", 2: "Extreme Heat"}
MENU_MAP = {0: "Basic", 1: "Special", 2: "Biryani"}
EVENT_MAP = {0: "Normal", 1: "Fest", 2: "Exam", 3: "Holiday"}
WEEKDAY_MAP = {0: "Monday", 1: "Tuesday", 2: "Wednesday", 3: "Thursday", 4: "Friday", 5: "Saturday", 6: "Sunday"}
MEAL_MAP = {0: "Breakfast", 1: "Lunch", 2: "Dinner"}

class SmartMessEnvironment:
    def __init__(self, task_level: str = "easy"):
        self.task_level = task_level
        self.max_steps = 30
        self.current_step = 0
        
        self.total_waste = 0.0
        self.total_shortage = 0.0
        self.current_obs = None

    def reset(self) -> SmartMessObservation:
        self.current_step = 0
        self.total_waste = 0.0
        self.total_shortage = 0.0
        self.current_obs = self._generate_next_observation()
        if os.path.exists("history_metadata.json"):
            os.remove("history_metadata.json")
        return self.current_obs

    def step(self, action: SmartMessAction) -> SmartMessObservation:
        self.current_step += 1
        
        weather_str = WEATHER_MAP.get(self.current_obs.weather, "Sunny")
        menu_str = MENU_MAP.get(self.current_obs.menu_type, "Basic")
        event_str = EVENT_MAP.get(self.current_obs.special_event, "Normal")
        weekday_str = WEEKDAY_MAP.get(self.current_obs.day_of_week, "Monday")
        
        forecast = predict_demand(
            registered_students=625,
            weather=weather_str,
            menu_type=menu_str,
            weekday=weekday_str,
            event=event_str
        )
        
        mu = forecast["probabilistic_forecast"]["mean_attendance"]
        sigma = forecast["probabilistic_forecast"]["std_dev"]
        
        actual_students = self._simulate_attendance(mu, sigma)
        
        meals_cooked = action.meals_to_prepare
        reward_val, waste, shortage, optimal_q, opt_res = self._calculate_reward(meals_cooked, actual_students, mu)
        
        self.total_waste += waste
        self.total_shortage += shortage
        done = self.current_step >= self.max_steps
        
        self._record_history(meals_cooked, actual_students, waste, shortage, mu, optimal_q, self.current_obs)

        next_obs = self._generate_next_observation()
        
        metadata = {
            "actual_students": actual_students,
            "forecast_mu": mu,
            "optimal_q": optimal_q,
            "shap_features": forecast["interpretability"]["feature_contributions"]
        }
        
        next_obs.reward = SmartMessReward(value=reward_val, metadata=metadata)
        next_obs.done = done
        next_obs.metadata = metadata
        
        self.current_obs = next_obs
        return next_obs

    def state(self) -> SmartMessState:
        return SmartMessState(
            step_count=self.current_step,
            total_waste=self.total_waste,
            total_shortage=self.total_shortage
        )

    def grade(self) -> float:
        worst_case_per_day = 500.0 * 1.0 
        worst_case_total = worst_case_per_day * self.max_steps
        
        total_penalty = (self.total_waste * 0.5) + (self.total_shortage * 1.0)
        raw_score = 1.0 - (total_penalty / worst_case_total)
        
        return float(round(max(0.0, min(1.0, raw_score)), 4))

    def _generate_next_observation(self) -> SmartMessObservation:
        obs = SmartMessObservation(
            day_of_week=random.randint(0, 6),
            meal_type=random.randint(0, 2),
            menu_type=random.choices([0, 1, 2], weights=[0.6, 0.2, 0.2])[0],
            weather=random.choices([0, 1, 2], weights=[0.7, 0.2, 0.1])[0],
            special_event=random.choices([0, 1, 2, 3], weights=[0.85, 0.05, 0.05, 0.05])[0]
        )
        obs.reward = SmartMessReward(value=0.0)
        return obs

    def _simulate_attendance(self, mu: float, sigma: float) -> float:
        if self.task_level == "easy":
            return mu
        if self.task_level == "medium":
            return mu + random.gauss(0, sigma * 0.5)
        if self.task_level == "hard":
            return mu + random.gauss(0, sigma)
        return mu

    def _calculate_reward(self, cooked: float, actual: float, mu: float) -> tuple[float, float, float, float, dict]:
        opt_res = optimize_cooking(
            food_prepared=int(cooked),
            actual_meals_served=int(actual),
            predicted_meals=int(mu),
            meal_type="Lunch",
            cost_per_meal_inr=50.0,
            shortage_probability_threshold=0.05
        )
        
        waste = float(opt_res["waste_summary"]["excess_meals"])
        shortage = max(0.0, actual - cooked)
        optimal_q = opt_res["stochastic_model"]["optimal_production_quantity_Q"]
        
        penalty = (waste * 0.5) + (shortage * 1.0)
        
        if cooked > optimal_q:
            penalty += (cooked - optimal_q) * 0.5
            
        reward_val = -penalty
        return reward_val, waste, shortage, optimal_q, opt_res

    def _record_history(self, cooked, actual, waste, shortage, mu, optimal_q, obs):
        history_path = "history_metadata.json"
        history = []
        if os.path.exists(history_path):
            try:
                with open(history_path, 'r') as f:
                    history = json.load(f)
            except:
                pass
                
        new_entry = {
            "step": self.current_step,
            "day": WEEKDAY_MAP.get(obs.day_of_week, "Monday"),
            "cooked": round(cooked, 2),
            "actual": round(actual, 2),
            "waste": round(waste, 2),
            "shortage": round(shortage, 2),
            "predicted_mu": round(mu, 2),
            "optimal_q": optimal_q,
            "task_level": self.task_level
        }
        history.append(new_entry)
        
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
