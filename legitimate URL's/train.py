import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Load the Majestic Million dataset
df = pd.read_csv('majestic_million.csv')

# 2. Preprocessing: Filter out new domains (no history) and sample 100,000 rows
df_clean = df[df['PrevGlobalRank'] != -1].copy()
df_sample = df_clean.sample(n=100000, random_state=42)

# Define Target: 1 if GlobalRank improved (decreased), 0 otherwise
df_sample['RankImproved'] = (df_sample['GlobalRank'] < df_sample['PrevGlobalRank']).astype(int)

# Features: Subnets and IPs (Current vs Previous)
features = ['RefSubNets', 'RefIPs', 'PrevRefSubNets', 'PrevRefIPs']
X = df_sample[features]
y = df_sample['RankImproved']

# 3. Split data (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Initialize and train the Best Model: Decision Tree
best_model = DecisionTreeClassifier(random_state=42)
best_model.fit(X_train, y_train)

# 5. Evaluate the model
y_pred = best_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"--- Best Model: Decision Tree ---")
print(f"Accuracy Score: {accuracy:.4f}")
print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Stayed/Dropped', 'Improved']))

# 6. Visualization: Confusion Matrix
plt.figure(figsize=(8, 6))
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Stayed/Dropped', 'Improved'], 
            yticklabels=['Stayed/Dropped', 'Improved'])
plt.xlabel('Predicted Label')
plt.ylabel('Actual Label')
plt.title('Confusion Matrix - Best Model (Decision Tree)')
plt.savefig('confusion_matrix_best_model.png')
print("\nConfusion Matrix image saved as 'confusion_matrix_best_model.png'")