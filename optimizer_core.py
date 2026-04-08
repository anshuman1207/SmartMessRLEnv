import math

def norm_pdf(x):
    return (1.0 / math.sqrt(2 * math.pi)) * math.exp(-0.5 * x**2)

def norm_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))

def get_z_score(service_level):
    if service_level >= 0.99: return 2.33
    if service_level >= 0.95: return 1.645
    if service_level >= 0.90: return 1.28
    if service_level >= 0.85: return 1.04
    if service_level >= 0.80: return 0.84
    return 0.0

def optimize_cooking(
    food_prepared: int, 
    actual_meals_served: int, 
    predicted_meals: int,
    meal_type: str,
    cost_per_meal_inr: float,
    shortage_probability_threshold: float = 0.05
) -> dict:
    """
    Stochastic Optimization Problem: Minimizes expected food waste subject to service-level constraints.
    """
    excess_meals = max(0, food_prepared - actual_meals_served)
    waste_percentage = round((excess_meals / food_prepared) * 100, 2) if food_prepared > 0 else 0
    
    mu = float(predicted_meals)
    sigma = max(1.0, mu * 0.10)
    
    service_level = 1.0 - shortage_probability_threshold
    z_score = get_z_score(service_level)
    
    optimal_q = math.ceil(mu + z_score * sigma)
    
    z_val = (optimal_q - mu) / sigma
    expected_waste_meals = (optimal_q - mu) * norm_cdf(z_val) + sigma * norm_pdf(z_val)
    
    return {
        "waste_summary": {
            "excess_meals": excess_meals,
            "waste_percentage": waste_percentage
        },
        "stochastic_model": {
            "optimal_production_quantity_Q": optimal_q,
            "expected_structural_waste_meals": round(expected_waste_meals, 2),
            "demand_mean_mu": mu,
            "demand_std_sigma": round(sigma, 2)
        }
    }
