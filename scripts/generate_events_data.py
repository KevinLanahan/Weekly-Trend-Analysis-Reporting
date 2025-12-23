import pandas as pd
import numpy as np

np.random.seed(42)

rows = []
departments = ["Dental", "Ortho", "Endo", "Perio"]
start_date = pd.to_datetime("2025-05-01")
end_date = pd.to_datetime("2025-08-31")

dates = pd.date_range(start_date, end_date, freq="D")

event_id = 1

for date in dates:
    # simulate 1–3 department entries per day
    daily_depts = np.random.choice(departments, size=np.random.randint(1, 4), replace=False)
    for dept in daily_depts:
        visits = np.random.randint(10, 80)
        revenue = visits * np.random.randint(80, 120)

        rows.append({
            "event_id": event_id,
            "event_date": date.strftime("%Y-%m-%d"),
            "department": dept,
            "visits": visits,
            "revenue": revenue
        })
        event_id += 1

df = pd.DataFrame(rows)

# Limit to ~180 rows so it's not massive
df = df.head(180)

df.to_csv("data/raw/events.csv", index=False)

print(f"Generated {len(df)} rows → data/raw/events.csv")
