import pandas as pd
import random
from datetime import datetime, timedelta

# Generate dates
start_date = datetime(2024, 1, 1)
daily_date = datetime(2025, 3, 23)

# Accounts
accounts = [f"ACC{i:03d}" for i in range(1, 11)]  # ACC001 to ACC010

# History (1,000 records)
history_data = []
for i in range(1000):
    tid = f"T{i+1:04d}"
    acc = random.choice(accounts)
    days_offset = random.randint(0, 364)  # Spread over 2024
    date = start_date + timedelta(days=days_offset)
    if acc == "ACC001":  # Upward trend
        base = 400 + (days_offset / 365) * 100  # 400 to 500
        balance = base + random.uniform(-10, 10)
    elif acc == "ACC002":  # Stable low
        balance = random.uniform(50, 60)
    else:  # Random for others
        balance = random.uniform(50, 2000)
    history_data.append([tid, acc, round(balance, 2), date.strftime('%Y-%m-%d')])

history_df = pd.DataFrame(history_data, columns=['Transaction ID', 'Account Number', 'Balance Difference', 'As of Date'])
history_df.to_csv('history.csv', index=False)
print("Generated history.csv with 1,000 records")

# Daily (100 records)
daily_data = []
for i in range(100):
    tid = f"T1{i+1:03d}"
    acc = random.choice(accounts)
    if acc == "ACC001":  # Upward trend continues
        base = 500 + (i % 8) * 10  # 500 to 570 over 8 rows
        balance = base + random.uniform(-5, 5)
    elif acc == "ACC002":  # Too high/too low
        balance = random.choice([random.uniform(50, 60), random.uniform(5000, 15000)])
    else:
        balance = random.uniform(50, 2000)
    daily_data.append([tid, acc, round(balance, 2), daily_date.strftime('%Y-%m-%d')])

daily_df = pd.DataFrame(daily_data, columns=['Transaction ID', 'Account Number', 'Balance Difference', 'As of Date'])
daily_df.to_csv('daily.csv', index=False)
print("Generated daily.csv with 100 records")