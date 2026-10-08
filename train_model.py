import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


np.random.seed(42)

n = 200

internal_marks = np.random.randint(20, 101, n)
attendance = np.random.randint(50, 101, n)
assignment_score = np.random.randint(30, 101, n)

result = (
    (0.5 * internal_marks +
     0.3 * attendance +
     0.2 * assignment_score) >= 60
).astype(int)

df = pd.DataFrame({
    "internal_marks": internal_marks,
    "attendance": attendance,
    "assignment_score": assignment_score,
    "result": result
})

X = df[[
    "internal_marks",
    "attendance",
    "assignment_score"
]]

y = df["result"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print(f"Accuracy: {accuracy:.4f}")

joblib.dump(model, "model.joblib")

with open("metrics.txt", "w") as f:
    f.write(f"accuracy={accuracy:.4f}\n")
