"""
Model training script for student admission prediction
This script loads processed data, trains a regression model, evaluates it, and saves to BentoML model store.
"""

import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime

# Machine learning libraries
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# BentoML for model storage
import bentoml

def load_processed_data():
    """Load the processed training and test datasets"""
    print("📁 Loading processed datasets...")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_path = os.path.join(project_root, 'data', 'processed')
    
    X_train = pd.read_csv(os.path.join(data_path, 'X_train.csv'))
    X_test = pd.read_csv(os.path.join(data_path, 'X_test.csv'))
    y_train = pd.read_csv(os.path.join(data_path, 'y_train.csv')).squeeze()
    y_test = pd.read_csv(os.path.join(data_path, 'y_test.csv')).squeeze()
    
    print(f"✅ Data loaded successfully!")
    print(f"Training set: X_train {X_train.shape}, y_train {y_train.shape}")
    print(f"Test set: X_test {X_test.shape}, y_test {y_test.shape}")
    print(f"Features: {list(X_train.columns)}")
    
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    """Train a linear regression model with feature scaling"""
    print(f"\n🔧 FEATURE SCALING AND MODEL TRAINING")
    print("=" * 50)
    
    # Initialize and fit the scaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    print("✅ Features scaled successfully!")
    
    # Initialize and train the model
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    
    print("✅ Linear Regression model trained successfully!")
    print(f"Model coefficients shape: {model.coef_.shape}")
    print(f"Model intercept: {model.intercept_:.6f}")
    
    # Display feature importance
    feature_importance = pd.DataFrame({
        'Feature': X_train.columns,
        'Coefficient': model.coef_,
        'Abs_Coefficient': np.abs(model.coef_)
    }).sort_values('Abs_Coefficient', ascending=False)
    
    print(f"\n📊 FEATURE IMPORTANCE:")
    print(feature_importance.to_string(index=False))
    
    return model, scaler, X_train_scaled

def evaluate_model(model, scaler, X_train, X_test, y_train, y_test, X_train_scaled):
    """Evaluate the trained model on training and test sets"""
    print(f"\n📈 MODEL EVALUATION")
    print("=" * 50)
    
    # Scale test data and make predictions
    X_test_scaled = scaler.transform(X_test)
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    
    # Calculate metrics for training set
    train_r2 = r2_score(y_train, y_train_pred)
    train_mse = mean_squared_error(y_train, y_train_pred)
    train_rmse = np.sqrt(train_mse)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    
    # Calculate metrics for test set
    test_r2 = r2_score(y_test, y_test_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_rmse = np.sqrt(test_mse)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    
    # Display results
    print("PERFORMANCE METRICS:")
    print(f"{'Metric':<12} {'Training':<12} {'Test':<12}")
    print("-" * 36)
    print(f"{'R² Score':<12} {train_r2:<12.6f} {test_r2:<12.6f}")
    print(f"{'MSE':<12} {train_mse:<12.6f} {test_mse:<12.6f}")
    print(f"{'RMSE':<12} {train_rmse:<12.6f} {test_rmse:<12.6f}")
    print(f"{'MAE':<12} {train_mae:<12.6f} {test_mae:<12.6f}")
    
    # Model interpretation
    print(f"\n🎯 MODEL INTERPRETATION:")
    print(f"• R² Score on test set: {test_r2:.4f} ({test_r2*100:.2f}% variance explained)")
    print(f"• RMSE on test set: {test_rmse:.6f} (average prediction error)")
    print(f"• The model explains {test_r2*100:.2f}% of the variance in admission chances")
    
    if test_r2 > 0.8:
        performance_level = "excellent"
        print("• 🟢 Excellent model performance!")
    elif test_r2 > 0.6:
        performance_level = "good"
        print("• 🟡 Good model performance")
    else:
        performance_level = "fair"
        print("• 🔴 Model needs improvement")
    
    return {
        'test_r2': test_r2,
        'test_rmse': test_rmse,
        'test_mae': test_mae,
        'train_r2': train_r2,
        'performance_level': performance_level
    }

def save_to_bentoml(model, scaler, X_train, metrics):
    """Save the trained model and scaler to BentoML model store"""
    print(f"\n🏪 SAVING TO BENTOML MODEL STORE")
    print("=" * 50)
    
    # Create metadata
    model_metadata = {
        "framework": "sklearn",
        "algorithm": "LinearRegression",
        "features": list(X_train.columns),
        "target": "Chance_of_Admit",
        "training_samples": len(X_train),
        "test_r2_score": float(metrics['test_r2']),
        "test_rmse": float(metrics['test_rmse']),
        "test_mae": float(metrics['test_mae']),
        "created_date": datetime.now().isoformat(),
        "description": "Linear regression model for predicting university admission chances"
    }
    
    try:
        # Save the trained model
        bento_model = bentoml.sklearn.save_model(
            "admission_predictor",
            model,
            metadata=model_metadata,
            labels={
                "type": "regression",
                "use_case": "admission_prediction",
                "performance": metrics['performance_level']
            }
        )
        
        print(f"✅ Model saved to BentoML store!")
        print(f"Model tag: {bento_model.tag}")
        
        # Save the scaler
        scaler_model = bentoml.sklearn.save_model(
            "admission_scaler",
            scaler,
            metadata={
                "description": "StandardScaler for admission prediction features",
                "features": list(X_train.columns),
                "created_date": datetime.now().isoformat()
            },
            labels={
                "type": "preprocessor",
                "use_case": "admission_prediction"
            }
        )
        
        print(f"✅ Scaler saved to BentoML store!")
        print(f"Scaler tag: {scaler_model.tag}")
        
        return bento_model.tag, scaler_model.tag
        
    except Exception as e:
        print(f"❌ Error saving to BentoML: {e}")
        return None, None

def save_backup_models(model, scaler, X_train):
    """Save backup models as joblib files"""
    print(f"\n💾 SAVING BACKUP MODELS")
    print("=" * 50)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    models_dir = os.path.join(project_root, 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    # Save models
    model_path = os.path.join(models_dir, 'admission_model.joblib')
    scaler_path = os.path.join(models_dir, 'scaler.joblib')
    features_path = os.path.join(models_dir, 'feature_names.joblib')
    
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(list(X_train.columns), features_path)
    
    print(f"✅ Backup models saved to {models_dir}/")

def main():
    """Main training pipeline"""
    print("🚀 STUDENT ADMISSION PREDICTION - MODEL TRAINING")
    print("=" * 60)
    
    # Load data
    X_train, X_test, y_train, y_test = load_processed_data()
    
    # Train model
    model, scaler, X_train_scaled = train_model(X_train, y_train)
    
    # Evaluate model
    metrics = evaluate_model(model, scaler, X_train, X_test, y_train, y_test, X_train_scaled)
    
    # Save to BentoML model store
    model_tag, scaler_tag = save_to_bentoml(model, scaler, X_train, metrics)
    
    # Save backup models
    save_backup_models(model, scaler, X_train)
    
    print(f"\n🎉 MODEL TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print(f"Model performance: {metrics['performance_level']}")
    print(f"Test R² Score: {metrics['test_r2']:.4f}")
    print(f"Test RMSE: {metrics['test_rmse']:.6f}")
    
    if model_tag:
        print(f"BentoML Model Tag: {model_tag}")
        print(f"BentoML Scaler Tag: {scaler_tag}")
    
    print("Ready for BentoML service creation! 🚀")

if __name__ == "__main__":
    main()
