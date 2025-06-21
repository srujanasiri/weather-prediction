import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
# Load your dataset
df = pd.read_excel("C:/datasets/climate1.xlsx")  
df=df.fillna(df.mean(numeric_only=True))# Change to your actual path
# Encode labels
le = LabelEncoder()
df['Condition'] = le.fit_transform(df['Condition'])

# Features and target
X = df[['Temperature(C)', 'Humidity(%)', 'Wind speed(km/h)', 'Precipitation(mm)']]
y = df['Condition']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression(max_iter=10000)
model.fit(X_train, y_train)

# Show accuracy
y_pred = model.predict(X_test)
print("Model Accuracy:", accuracy_score(y_test, y_pred))

# --- User input at runtime ---
try:
    temp = float(input("Enter Temperature: "))
    humidity = float(input("Enter Humidity: "))
    wind = float(input("Enter Wind Speed: "))
    precip = float(input("Enter Precipitation: "))

    # Create DataFrame for prediction
    user_data = pd.DataFrame({
        'Temperature(C)': [temp],
        'Humidity(%)': [humidity],
        'Wind speed(km/h)': [wind],
        'Precipitation(mm)': [precip]
    })

    prediction = model.predict(user_data)
    result = le.inverse_transform(prediction)[0]
    print("Predicted Weather Condition:", result)
except Exception as e:
    print("error:",e)
