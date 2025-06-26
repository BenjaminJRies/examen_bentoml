"""
Test script to verify BentoML integration and model training
"""

import os
import sys

# Add the project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("🧪 Testing BentoML Integration")
print("=" * 40)

try:
    import bentoml
    print(f"✅ BentoML imported successfully - Version: {bentoml.__version__}")
except ImportError as e:
    print(f"❌ Failed to import BentoML: {e}")
    sys.exit(1)

try:
    import pandas as pd
    import numpy as np
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    print("✅ All required libraries imported successfully")
except ImportError as e:
    print(f"❌ Failed to import required libraries: {e}")
    sys.exit(1)

# Test data loading
try:
    print("\n📁 Testing data loading...")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_path = os.path.join(project_root, 'data', 'processed')
    
    X_train = pd.read_csv(os.path.join(data_path, 'X_train.csv'))
    X_test = pd.read_csv(os.path.join(data_path, 'X_test.csv'))
    y_train = pd.read_csv(os.path.join(data_path, 'y_train.csv')).squeeze()
    y_test = pd.read_csv(os.path.join(data_path, 'y_test.csv')).squeeze()
    
    print(f"✅ Data loaded: Train {X_train.shape}, Test {X_test.shape}")
    
except Exception as e:
    print(f"❌ Data loading failed: {e}")
    sys.exit(1)

# Test model training
try:
    print("\n🤖 Testing model training...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    
    y_test_pred = model.predict(X_test_scaled)
    r2 = r2_score(y_test, y_test_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    
    print(f"✅ Model trained successfully")
    print(f"   R² Score: {r2:.4f}")
    print(f"   RMSE: {rmse:.6f}")
    
except Exception as e:
    print(f"❌ Model training failed: {e}")
    sys.exit(1)

# Test BentoML model saving
try:
    print("\n🏪 Testing BentoML model saving...")
    
    # Save model to BentoML store
    bento_model = bentoml.sklearn.save_model(
        "admission_predictor_test",
        model,
        metadata={
            "framework": "sklearn",
            "algorithm": "LinearRegression",
            "test_r2_score": float(r2),
            "test_rmse": float(rmse)
        }
    )
    
    print(f"✅ Model saved to BentoML store!")
    print(f"   Model tag: {bento_model.tag}")
    
    # Save scaler
    scaler_model = bentoml.sklearn.save_model(
        "admission_scaler_test", 
        scaler,
        metadata={"description": "StandardScaler for admission prediction"}
    )
    
    print(f"✅ Scaler saved to BentoML store!")
    print(f"   Scaler tag: {scaler_model.tag}")
    
except Exception as e:
    print(f"❌ BentoML saving failed: {e}")
    print(f"   Error details: {str(e)}")

print("\n🎉 All tests completed!")
