import os
import json
import asyncio
from openai import OpenAI
import sys

from environment import SmartMessEnvironment
from models import SmartMessAction

def run_inference():
    api_base_url = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
    model_name = os.getenv("MODEL_NAME", "gpt-4o")
    api_key = os.getenv("HF_TOKEN", "mock_key")
    
    client = OpenAI(
        base_url=api_base_url,
        api_key=api_key
    )
    
    for task_level in ["easy", "medium", "hard"]:
        env = SmartMessEnvironment(task_level=task_level)
        obs = env.reset()
        done = False
        
        # REQUIRED FORMAT: [START]
        print(f"[START] task={task_level} env=SmartMess model={model_name}", flush=True)
        
        total_rewards = 0.0
        steps_taken = 0
        success = False
        
        while not done:
            steps_taken += 1
            
            prompt = f"""
            You are an AI managing a mess hall. Predict how many students will eat out of an expected base.
            Observation mappings:
            - Weather: 0=Sunny (x1.05), 1=Rainy (x0.85), 2=Heat (x0.90)
            - Menu: 0=Basic (x0.95), 1=Special (x1.20), 2=Biryani (x1.35)
            - Day: 0-4=Weekday (x1.05), 5-6=Weekend (x0.85)
            - Event: 0=Normal (x1.0), 1=Fest (x1.25), 2=Exam (x0.80), 3=Holiday (x0.60)
            
            The baseline registered students is roughly 500 when all multipliers are 1.0.

            CURRENT OBSERVATION:  
            Day: {obs.day_of_week}, Weather: {obs.weather}, Menu: {obs.menu_type}, Event: {obs.special_event}
            
            Respond with raw JSON only containing "meals_to_prepare" (float).
            Example: {{"meals_to_prepare": 510.0}}
            """
            
            try:
                # Enforcing temperature 0.0 and seed for exact reproducibility
                res = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                    temperature=0.0,
                    seed=42 
                )
                
                ai_choice = json.loads(res.choices[0].message.content)
                action_value = float(ai_choice.get("meals_to_prepare", 500))
                action = SmartMessAction(meals_to_prepare=action_value)
                error = None
            except Exception as e:
                # If the AI breaks JSON formatting, fallback to baseline and log error
                action_value = 500.0
                action = SmartMessAction(meals_to_prepare=action_value)
                error = str(e).replace(' ', '_')
                
            # Push the AI's action into the environment
            obs = env.step(action)
            done = obs.done
            reward = obs.reward.value
            total_rewards += reward
            
            # REQUIRED FORMAT: [STEP]
            error_str = error if error else "None"
            print(f"[STEP] step={env.current_step} action={action_value} reward={reward:.2f} done={done} error={error_str}", flush=True)

        state = env.state()
        score = env.grade()
        
        # SUCCESS_SCORE_THRESHOLD
        success_threshold = 0.5
        success = score >= success_threshold

        # REQUIRED FORMAT: [END]
        print(f"[END] success={success} steps={env.current_step} score={score:.4f} rewards={total_rewards:.2f}", flush=True)

if __name__ == "__main__":
    run_inference()
