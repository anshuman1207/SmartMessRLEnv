# SmartMess RL 🍲

An OpenEnv-compliant Reinforcement Learning environment designed to predict and manage college mess food preparation, minimizing waste while preventing student shortages.

## 🎯 Problem Description
Every day, university mess halls struggle with forecasting attendance. If they cook too much food, thousands of dollars are wasted in the trash. If they cook too little, students go hungry, leading to poor satisfaction and complaints. 

**SmartMess RL** challenges AI agents to act as the Mess Hall Manager. The AI must learn to interpret context (the day of the week, weather, and campus events) and predict precisely how many meals to cook for the day.

## 🧠 The Environment

### Observation (What the AI sees)
The AI is provided context before every meal.
*   `day_of_week` (0-6): Monday through Sunday.
*   `meal_type` (0-2): Breakfast, Lunch, or Dinner.
*   `weather` (0-1): Sunny or Rainy/Snowy.
*   `special_event` (0-1): Normal day or Campus Festival.

### Action (The AI's decision)
*   `meals_to_prepare` (float): The exact number of plates the AI chooses to cook based on the observation.

### Reward (How the AI learns)
The AI receives continuous feedback with asymmetric penalties (indicating that hunger is much worse than waste!):
*   **Waste Penalty:** -0.5 points per excess meal cooked. 
*   **Shortage Penalty:** -1.0 points per hungry student.

The maximum possible reward is `0.0`. The AI loses points for every mistake.

## 🏋️ Tasks (Difficulty Levels)
The environment can be initialized with three distinct difficulties:
1.  **Easy:** Completely predictable attendance based solely on whether it is a weekday or weekend.
2.  **Medium:** Applies human unpredictability (noise) of +/- 20 students to the standard baseline.
3.  **Hard:** Evaluates complex causality. Rain heavily decreases attendance, while campus events heavily increase attendance. Also includes aggressive noise.

## 🚀 Setup Instructions
Make sure you have Docker installed, then run the following in this directory:

```bash
# 1. Build the lightweight container
docker build -t smartmess-env .

# 2. Run the environment, passing in your API key dynamically!
docker run -e HF_TOKEN="your_key_here" smartmess-env
```

## 🏆 Leaderboard & Baseline
We have provided a hard-coded heuristic baseline agent (`baseline.py`) that successfully identifies weather patterns and intelligently biases slightly toward overcooking to mitigate the harsh shortage penalties.

**Baseline Score (Hard Task):** `0.9691 / 1.0`  

*Can your Large Language Model beat the baseline?*
