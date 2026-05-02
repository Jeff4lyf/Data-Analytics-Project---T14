import pandas as pd

def croston_method(data, alpha):
    n = len(data)
    
    demand_level = []
    interarrival_time = []
    forecasts = [0.0] * (n + 1)
    
    first_non_zero_idx = -1
    for i in range(n):
        if data[i] != 0:
            first_non_zero_idx = i
            break
            
    if first_non_zero_idx == -1:
        return [0.0] * (n + 1)
        
    first_demand = data[first_non_zero_idx]
    E = first_demand
    P = first_non_zero_idx + 1
    
    demand_level.append(E)
    interarrival_time.append(P)
    
    if P != 0:
        f = E / P
    else:
        f = 0.0
        
    for i in range(first_non_zero_idx + 1, n + 1):
        forecasts[i] = f
        
    q = 0
    last_demand_idx = first_non_zero_idx
    
    print(f"Initialization:")
    print(f"  First demand at period {first_non_zero_idx+1}: {first_demand}")
    print(f"  Initial E: {E}, Initial P: {P}, Initial Forecast (starting P{first_non_zero_idx+2}): {f:.2f}")
    print("-" * 20)
    
    for i in range(first_non_zero_idx + 1, n):
        d_curr = data[i]
        q += 1
        
        if d_curr != 0:
            E = alpha * d_curr + (1 - alpha) * E
            P = alpha * q + (1 - alpha) * P
            
            demand_level.append(E)
            interarrival_time.append(P)
            
            if P != 0:
                f = E / P
            else:
                f = 0.0
            
            print(f"Demand at P{i+1}: {d_curr}, periods since last: {q}")
            print(f"  Updated E: {E:.2f}, Updated P: {P:.2f}, Updated Forecast (starting P{i+2}): {f:.2f}")
            print("-" * 20)
            
            if i + 1 <= n:
                forecasts[i+1] = f
            q = 0
        else:
            if i + 1 <= n:
                forecasts[i+1] = forecasts[i]

    return forecasts

demand_data = [0, 5, 0, 0, 8, 0, 0, 0, 10, 0]
alpha_param = 0.2

print(f"Croston's Method Forecasting")
print(f"Demand Data: {demand_data}")
print(f"Alpha: {alpha_param}\n")

all_forecasts = croston_method(demand_data, alpha_param)

results_df = pd.DataFrame({
    'Period': range(1, len(demand_data) + 1),
    'Demand': demand_data,
    'Forecast': all_forecasts[:-1]
})

print(f"\nDemand and Forecast for each period:")
print(results_df)

final_forecast = all_forecasts[-1]
print(f"\nForecast for Period {len(demand_data) + 1}: {final_forecast:.2f}")
