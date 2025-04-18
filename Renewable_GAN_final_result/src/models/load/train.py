import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import holidays
from datetime import datetime
import os
import joblib

# Set random seed for reproducibility
np.random.seed(42)

# if model already exists, load it
if os.path.exists('models/load/saved_models/xgboost_load_forecasting_optimized.model'):
    print("Loading existing model...")
    final_model = xgb.Booster()
    final_model.load_model('models/load/saved_models/xgboost_load_forecasting_optimized.model')
else:
    
    data = pd.read_csv("models/load/dataset_for_load.csv")
    data['date_time'] = pd.to_datetime(data['date_time'])

    # Add time-based features
    data['hour'] = data['date_time'].dt.hour
    data['day_of_week'] = data['date_time'].dt.dayofweek
    data['month'] = data['date_time'].dt.month
    data['is_weekend'] = (data['day_of_week'] >= 5).astype(int)
    data['day_of_year'] = data['date_time'].dt.dayofyear
    data['week_of_year'] = data['date_time'].dt.isocalendar().week

    # Add Singapore holiday feature
    sg_holidays = holidays.SG()  # Use Singapore holidays
    data['is_holiday'] = data['date_time'].dt.date.apply(lambda x: x in sg_holidays).astype(int)

    # Add cyclical encoding for time features
    def cyclical_encoding(df, col, max_val):
        df[f'{col}_sin'] = np.sin(2 * np.pi * df[col]/max_val)
        df[f'{col}_cos'] = np.cos(2 * np.pi * df[col]/max_val)
        return df

    data = cyclical_encoding(data, 'hour', 24)
    data = cyclical_encoding(data, 'day_of_week', 7)
    data = cyclical_encoding(data, 'month', 12)
    data = cyclical_encoding(data, 'day_of_year', 365)

    # Select features including new time-based features
    features = data[['nwp_globalirrad', 'nwp_directirrad', 'nwp_temperature', 'nwp_humidity', 
                    'nwp_windspeed', 'lmd_totalirrad', 'lmd_diffuseirrad', 'lmd_temperature', 
                    'lmd_pressure', 'hour', 'day_of_week', 'month', 'is_weekend', 'is_holiday',
                    'hour_sin', 'hour_cos', 'day_of_week_sin', 'day_of_week_cos',
                    'month_sin', 'month_cos', 'day_of_year_sin', 'day_of_year_cos']]
                    
    target = data[['LOAD']]

    # Use StandardScaler
    scaler_feat = StandardScaler()
    scaler_load = StandardScaler()

    scaled_features = scaler_feat.fit_transform(features)
    scaled_load = scaler_load.fit_transform(target)

    # Save the scaler
    
    joblib.dump(scaler_feat, '../../models/load/saved_models/scaler_features.pkl')
    joblib.dump(scaler_load, '../../models/load/saved_models/scaler_load.pkl')

    print("No existing model found. Training a new model...")
    # Train XGBoost model with early stopping

    # Create feature dataframe for XGBoost (not required to use scaled features)
    X = pd.DataFrame(data=scaled_features, columns=features.columns)
    y = pd.DataFrame(data=scaled_load, columns=['LOAD'])

    # Time series split instead of random split
    tscv = TimeSeriesSplit(n_splits=5)
    for train_index, test_index in tscv.split(X):
        X_train, X_val = X.iloc[train_index], X.iloc[test_index]
        y_train, y_val = y.iloc[train_index], y.iloc[test_index]
        # Use the last split

    print(f"Train set size: {X_train.shape[0]}, Validation set size: {X_val.shape[0]}")

    # Calculate evaluation metrics
    def calculate_metrics(actual, predicted):
        actual = np.array(actual)
        predicted = np.array(predicted)
        
        # Mean Absolute Error
        mae = mean_absolute_error(actual, predicted)
        
        # Root Mean Squared Error
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        
        # Mean Absolute Percentage Error
        mape = mean_absolute_percentage_error(actual, predicted) * 100
        
        return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}

    # XGBoost model parameters
    xgb_params = {
        'objective': 'reg:squarederror',
        'booster': 'gbtree',
        'eta': 0.05,
        'max_depth': 8,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 3,
        'gamma': 0.1,
        'alpha': 0.2,
        'lambda': 1.0,
        'tree_method': 'hist',
        'seed': 42
    }

    # Create DMatrix for XGBoost training
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    # Set up evaluation list
    watchlist = [(dtrain, 'train'), (dval, 'eval')]
    print("Training XGBoost model...")
    model = xgb.train(
        params=xgb_params,
        dtrain=dtrain,
        num_boost_round=2000,
        evals=watchlist,
        early_stopping_rounds=50,
        verbose_eval=50
    )

    # Save the model
    model.save_model('xgboost_load_forecasting.model')
    print("Model saved to xgboost_load_forecasting.model")

    # Make predictions on validation set
    y_pred = model.predict(dval)

    # Inverse transform predictions and actual values to original scale
    y_val_original = scaler_load.inverse_transform(y_val)
    y_pred_original = scaler_load.inverse_transform(y_pred.reshape(-1, 1))

    # Calculate metrics
    final_metrics = calculate_metrics(y_val_original, y_pred_original)
    print("\nFinal Validation Metrics:")
    print(f"MAE: {final_metrics['MAE']:.4f}")
    print(f"RMSE: {final_metrics['RMSE']:.4f}")
    print(f"MAPE: {final_metrics['MAPE']:.2f}%")


    # Get corresponding dates for validation set
    val_dates = data['date_time'].iloc[-len(y_val):]


    # Hyperparameter tuning with optuna
    import optuna

    def objective(trial):
        params = {
            'objective': 'reg:squarederror',
            'booster': trial.suggest_categorical('booster', ['gbtree']),
            'eta': trial.suggest_float('eta', 0.01, 0.2),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 0, 1),
            'alpha': trial.suggest_float('alpha', 0, 1),
            'lambda': trial.suggest_float('lambda', 0.1, 2.0),
            'seed': 42
        }
        
        pruning_callback = optuna.integration.XGBoostPruningCallback(trial, "eval-rmse")
        
        model = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=1000,
            evals=watchlist,
            early_stopping_rounds=50,
            verbose_eval=False,
            callbacks=[pruning_callback]
        )
        
        preds = model.predict(dval)
        rmse = np.sqrt(mean_squared_error(y_val, preds))
        return rmse

    # Comment out if you don't want to run hyperparameter tuning
    print("\nRunning hyperparameter optimization (this may take some time)...")
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=20)  # Use more trials for better results

    print("\nBest hyperparameters:")
    print(study.best_params)

    # Train final model with best parameters
    best_params = study.best_params
    best_params['objective'] = 'reg:squarederror'
    best_params['seed'] = 42

    print("\nTraining final model with best parameters...")
    final_model = xgb.train(
        params=best_params,
        dtrain=dtrain,
        num_boost_round=2000,
        evals=watchlist,
        early_stopping_rounds=50,
        verbose_eval=50
    )

    # Save the final model
    final_model.save_model('xgboost_load_forecasting_optimized.model')
    print("Optimized model saved to xgboost_load_forecasting_optimized.model")

    # Make predictions with optimized model
    y_pred_best = final_model.predict(dval)
    y_pred_best_original = scaler_load.inverse_transform(y_pred_best.reshape(-1, 1))

    # Calculate metrics for optimized model
    best_metrics = calculate_metrics(y_val_original, y_pred_best_original)
    print("\nOptimized Model Validation Metrics:")
    print(f"MAE: {best_metrics['MAE']:.4f}")
    print(f"RMSE: {best_metrics['RMSE']:.4f}")
    print(f"MAPE: {best_metrics['MAPE']:.2f}%")

    # Plot date vs actual and predicted load values based on optimized model on 100 values
    plt.figure(figsize=(15, 6))
    plt.plot(val_dates[:300], y_val_original[:300], label='Actual Load', color='black')
    plt.plot(val_dates[:300], y_pred_best_original[:300], label='Predicted Load', color='red')
    plt.title('Load Forecasting: Actual vs Predicted (Optimized Model)')
    plt.xlabel('Date')
    plt.ylabel('Load')
    plt.legend()
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=1))
    plt.gcf().autofmt_xdate()
    plt.grid()
    plt.savefig('load_forecasting_optimized_300.png')


    # Predict on all values in the dataset and save the results under the name 'predicted_load' fields in the dataset

    print("\nPredicting on all values in the dataset...")
    # Create DMatrix for the entire dataset
    dall = xgb.DMatrix(X)
    # Make predictions on the entire dataset
    y_all_pred = final_model.predict(dall)
    # Inverse transform predictions to original scale
    y_all_pred_original = scaler_load.inverse_transform(y_all_pred.reshape(-1, 1))

    data['predicted_load'] = y_all_pred_original

    data.drop(columns=['hour', 'day_of_week', 'month', 'is_weekend', 'is_holiday',
                    'hour_sin', 'hour_cos', 'day_of_week_sin', 'day_of_week_cos',
                    'month_sin', 'month_cos', 'day_of_year_sin', 'day_of_year_cos', 'day_of_year', 'week_of_year'], inplace=True)

    # Save the updated dataframe to a new CSV file
    output_file = 'predicted_load_dataset_final.csv'
    data.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")


def load_model(input_file):
    """
    Load the trained XGBoost model and predict load for a new dataset.
    Args:
        input_file (str): Path to the new input CSV file.
    """
    new_data = pd.read_csv(input_file)
    new_data['date_time'] = pd.to_datetime(new_data['date_time'])

    # Add same time-based features
    new_data['hour'] = new_data['date_time'].dt.hour
    new_data['day_of_week'] = new_data['date_time'].dt.dayofweek
    new_data['month'] = new_data['date_time'].dt.month
    new_data['is_weekend'] = (new_data['day_of_week'] >= 5).astype(int)
    new_data['day_of_year'] = new_data['date_time'].dt.dayofyear
    new_data['week_of_year'] = new_data['date_time'].dt.isocalendar().week

    # Add Singapore holiday feature
    sg_holidays = holidays.SG()
    new_data['is_holiday'] = new_data['date_time'].dt.date.apply(lambda x: x in sg_holidays).astype(int)

    def cyclical_encoding(df, col, max_val):
        df[f'{col}_sin'] = np.sin(2 * np.pi * df[col]/max_val)
        df[f'{col}_cos'] = np.cos(2 * np.pi * df[col]/max_val)
        return df
    # Add cyclical encoding
    new_data = cyclical_encoding(new_data, 'hour', 24)
    new_data = cyclical_encoding(new_data, 'day_of_week', 7)
    new_data = cyclical_encoding(new_data, 'month', 12)
    new_data = cyclical_encoding(new_data, 'day_of_year', 365)


    # Select the same features as training
    features_new = new_data[['nwp_globalirrad', 'nwp_directirrad', 'nwp_temperature', 'nwp_humidity', 
                             'nwp_windspeed', 'lmd_totalirrad', 'lmd_diffuseirrad', 'lmd_temperature', 
                             'lmd_pressure', 'hour', 'day_of_week', 'month', 'is_weekend', 'is_holiday',
                             'hour_sin', 'hour_cos', 'day_of_week_sin', 'day_of_week_cos',
                             'month_sin', 'month_cos', 'day_of_year_sin', 'day_of_year_cos']]

    scaler_feat = joblib.load('models/load/saved_models/scaler_features.pkl')
    scaler_load = joblib.load('models/load/saved_models/scaler_load.pkl')

    # Scale the features
    scaled_features_new = scaler_feat.transform(features_new)


    X = pd.DataFrame(data=scaled_features_new , columns=features_new.columns)

    # Create DMatrix for XGBoost
    dnew = xgb.DMatrix(X)

    # Load the optimized trained model
    model = xgb.Booster()
    model.load_model('models/load/saved_models/xgboost_load_forecasting_optimized.model')

    # Make predictions
    y_new_pred = model.predict(dnew)

    # Inverse transform predictions to original scale
    y_new_pred = scaler_load.inverse_transform(y_new_pred.reshape(-1, 1))

    # Add predictions to new_data
    new_data['predicted_load'] = y_new_pred

    # Drop extra columns to clean up
    new_data.drop(columns=['hour', 'day_of_week', 'month', 'is_weekend', 'is_holiday',
                           'hour_sin', 'hour_cos', 'day_of_week_sin', 'day_of_week_cos',
                           'month_sin', 'month_cos', 'day_of_year_sin', 'day_of_year_cos', 'day_of_year', 'week_of_year'], inplace=True)

   
    return new_data





