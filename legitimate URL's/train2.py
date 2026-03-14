import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Load your dataset
df = pd.read_csv('Phising_Detection_Dataset_Converted (1).csv')

# 2. Define Features (X) and Target (y)
# Features based on your CSV: url_length, num_subdomains, has_ip_in_url, etc.
X = df.drop('is_phishing', axis=1) 
y = df['is_phishing']

# 3. Split the data into Training and Testing sets (e.g., 80/20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Initialize and Train the Random Forest Classifier
# n_estimators=100 creates 100 decision trees (the "Forest")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. Make Predictions
y_pred = model.predict(X_test)

# 6. Evaluate the Model
print(f"Accuracy Score: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))