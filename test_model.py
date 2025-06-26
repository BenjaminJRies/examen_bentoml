"""
Test script to verify the model training pipeline works correctly
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import joblib
import os

def main():
    print("🧪 Testing Model Training Pipeline")
    print("=" * 50)
    
    # Load processed data
    print("Loading processed data...")
    X_train = pd.read_csv('data/processed/X_train.csv')
    X_test = pd.read_csv('data/processed/X_test.csv')
    y_train = pd.read_csv('data/processed/y_train.csv').squeeze()
    y_test = pd.read_csv('data/processed/y_test.csv').squeeze()
    
    print(f"✅ Data loaded: {X_train.shape} train, {X_test.shape} test")
    
    # Scale features
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("✅ Features scaled")
    
    # Train model
    print("Training linear regression model...")
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    print("✅ Model trained")
    
    # Evaluate model
    print("Evaluating model...")
    y_test_pred = model.predict(X_test_scaled)
    r2 = r2_score(y_test, y_test_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    
    print(f"📊 Test R² Score: {r2:.4f}")
    print(f"📊 Test RMSE: {rmse:.6f}")
    
    # Save model
    print("Saving model...")
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/admission_model.joblib')
    joblib.dump(scaler, 'models/scaler.joblib')
    joblib.dump(list(X_train.columns), 'models/feature_names.joblib')
    print("✅ Model saved")
    
    # Test prediction
    print("Testing prediction...")
    sample = X_test_scaled[:1]
    prediction = model.predict(sample)[0]
    actual = y_test.iloc[0]
    print(f"📈 Sample prediction: {prediction:.4f} (actual: {actual:.4f})")
    
    print("\n🎉 Model training pipeline test completed successfully!")
    
    return {
        'r2_score': r2,
        'rmse': rmse,
        'model_saved': True
    }

if __name__ == "__main__":
    results = main()
