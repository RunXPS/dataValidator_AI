from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pandas as pd

# Load and preprocess
df = pd.read_csv("entrepreneurs.csv")
df['business_from_major'] = (df['Major'] == df['Business Industry']).astype(int)

# Encode categorical vars
df = pd.get_dummies(df, columns=['Gender', 'Major', 'School'])

# Features and labels
X = df.drop(columns=['Name', 'Has Business', 'Business Industry', 'business_from_major'])
y = df['business_from_major']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
