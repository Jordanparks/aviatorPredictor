import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# Example dataset
data = {'Time': ['22:00', '22:10', '22:20'],
        'Multiplier': [1.5, 2.1, 1.8]}

df = pd.DataFrame(data)
df.to_csv("aviator_data.csv", index=False)

# Train a simple prediction model
X = np.arange(len(df)).reshape(-1, 1)
y = df['Multiplier']
model = RandomForestRegressor(n_estimators=100)
model.fit(X, y)

# Predict next multiplier
next_prediction = model.predict([[len(df) + 1]])
print(f"🔮 Predicted Next Multiplier: {next_prediction[0]:.2f}")
