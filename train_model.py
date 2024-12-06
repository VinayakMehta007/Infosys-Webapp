import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import joblib

# Function to train and save the model
def train_model():
    # Load the dataset
    data = pd.read_csv("healthcare-dataset-stroke-data (2).csv")  # Update with the correct path

    # Handle missing values
    # Impute missing numerical values with the mean
    numerical_columns = data.select_dtypes(include=['float64', 'int64']).columns
    imputer = SimpleImputer(strategy="mean")
    data[numerical_columns] = imputer.fit_transform(data[numerical_columns])

    # Impute missing categorical values with the most frequent value
    categorical_columns = data.select_dtypes(include=['object']).columns
    cat_imputer = SimpleImputer(strategy="most_frequent")
    data[categorical_columns] = cat_imputer.fit_transform(data[categorical_columns])

    # Encode categorical variables
    label_encoders = {}
    for col in categorical_columns:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        label_encoders[col] = le  # Save encoders for future use (if needed)

    # Separate features (X) and target (y)
    X = data.drop(columns=["stroke", "id"])  # Drop 'id' and 'stroke' columns
    y = data["stroke"]

    # Train-Test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train the Logistic Regression model
    model = LogisticRegression(random_state=42)
    model.fit(X_train_scaled, y_train)

    # Save the model, scaler, and label encoders
    joblib.dump(model, "logistic_model.pkl")
    joblib.dump(scaler, "scaler.pkl")
    joblib.dump(label_encoders, "label_encoders.pkl")  # Save encoders if needed
    
    # Save the feature names used in training
    feature_names = X.columns.tolist()  # Store the feature names
    joblib.dump(feature_names, "feature_names.pkl")

    print("Model, Scaler, Encoders, and Feature Names saved successfully!")

# Run the training function if this is the main script
if __name__ == "__main__":
    train_model()
