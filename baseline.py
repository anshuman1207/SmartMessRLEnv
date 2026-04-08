from models import SmartMessObservation, SmartMessAction
from environment import SmartMessEnvironment

class BaselineAgent:
    """A deterministic heuristic agent to act as a baseline benchmark."""
    
    def predict(self, obs: SmartMessObservation) -> SmartMessAction:
        base_expected = 625 * 0.8  # 500

        mu = base_expected
        
        mu *= {0: 1.05, 1: 0.85, 2: 0.90}.get(obs.weather, 1.0)

        mu *= {0: 0.95, 1: 1.20, 2: 1.35}.get(obs.menu_type, 1.0)

        if obs.day_of_week >= 5:
            mu *= 0.85
        else:
            mu *= 1.05

        mu *= {0: 1.0, 1: 1.25, 2: 0.80, 3: 0.60}.get(obs.special_event, 1.0)
        
        sigma = mu * 0.10
        
        z_score = 0.43 
        
        optimal_q = mu + (z_score * sigma)
            
        return SmartMessAction(meals_to_prepare=optimal_q)

if __name__ == "__main__":
    env = SmartMessEnvironment(task_level="hard")
    agent = BaselineAgent()
    
    obs = env.reset()
    done = False
    
    print("\n[SMARTMESS BASELINE AGENT]")
    print("Testing on Hard difficulty...")
    
    while not done:
        action = agent.predict(obs)
        obs = env.step(action)
        done = obs.done
        
    state = env.state()
    print(f"Episode Finished! ({state.step_count} days)")
    print(f"Total Waste: {state.total_waste:.1f} meals")
    print(f"Total Shortage: {state.total_shortage:.1f} meals")
    print(f"Final Leaderboard Grade: {env.grade()}/1.0")
    print("-" * 30 + "\n")
