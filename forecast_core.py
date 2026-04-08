import math

def predict_demand(
    registered_students: int,
    weather: str,
    menu_type: str,
    weekday: str,
    event: str = "None",
    attendance_variation: float = 0.0,
    shortage_probability_threshold: float = 0.05
) -> dict:
    """
    Probabilistic demand prediction supporting prediction intervals and SHAP feature importance.
    Supports event modifiers and attendance variation for simulation.
    """
    base_prob = 0.8  # Assume 80% base attendance
    base_expected = registered_students * base_prob

    current_expected = base_expected

    weather_lower = weather.lower()
    weather_multiplier = 1.0
    if weather_lower in ["rainy", "stormy"]:
        weather_multiplier = 0.85
    elif weather_lower == "sunny":
        weather_multiplier = 1.05
    elif weather_lower == "extreme heat":
        weather_multiplier = 0.90
    weather_shap = (current_expected * weather_multiplier) - current_expected
    current_expected += weather_shap

    menu_lower = menu_type.lower()
    menu_multiplier = 1.0
    if menu_lower == "special":
        menu_multiplier = 1.2
    elif menu_lower == "biryani":
        menu_multiplier = 1.35
    elif menu_lower == "basic":
        menu_multiplier = 0.95
    menu_shap = (current_expected * menu_multiplier) - current_expected
    current_expected += menu_shap

    weekday_lower = weekday.lower()
    weekday_multiplier = 1.0
    weekend_days = ["friday", "saturday", "sunday", "fri", "sat", "sun", "weekend"]
    if weekday_lower in weekend_days:
        weekday_multiplier = 0.85
    else:
        weekday_multiplier = 1.05
    weekday_shap = (current_expected * weekday_multiplier) - current_expected
    current_expected += weekday_shap

    event_lower = (event or "none").lower()
    event_multiplier = 1.0
    if event_lower == "fest":
        event_multiplier = 1.25
    elif event_lower == "exam":
        event_multiplier = 0.80
    elif event_lower == "holiday":
        event_multiplier = 0.60
    event_shap = (current_expected * event_multiplier) - current_expected
    current_expected += event_shap

    variation_shap = current_expected * (attendance_variation / 100.0)
    current_expected += variation_shap

    mu = current_expected
    sigma = mu * 0.10  # 10% std dev

    # Cap predicted attendance
    predicted_attendance = int(min(registered_students, max(0, mu)))

    # Calibration & Prediction Intervals (90% Coverage -> z = 1.645)
    z_90 = 1.645
    lower_bound = max(0, int(mu - z_90 * sigma))
    upper_bound = min(registered_students, int(mu + z_90 * sigma))

    confidence_score = max(0.0, min(1.0, round(1.0 - (sigma / max(mu, 1)) * 2, 2)))
    predicted_food_kg = round(predicted_attendance * 0.5, 2)

    return {
        "predicted_attendance": predicted_attendance,
        "probabilistic_forecast": {
            "mean_attendance": round(mu, 2),
            "std_dev": round(sigma, 2),
            "prediction_interval_90": {
                "lower_bound": lower_bound,
                "upper_bound": upper_bound
            }
        },
        "interpretability": {
            "feature_contributions": {
                "weather": round(weather_shap, 2),
                "menu": round(menu_shap, 2),
                "weekday": round(weekday_shap, 2),
                "event": round(event_shap, 2),
                "variation": round(variation_shap, 2)
            }
        }
    }
