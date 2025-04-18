import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import optuna

# Set random seed
np.random.seed(42)


if os.path.exists('models/usep/saved_models/best_lgb_model.pkl'):
    print("Model already exists. Skipping training. Loading the model...")

else:
    print("Model does not exist. Training the model...")

    # Load and preprocess data
    df = pd.read_csv('predicted_load_dataset_final.csv')
    df['date_time'] = pd.to_datetime(df['date_time'])

    # Feature Engineering
    df['hour'] = df['date_time'].dt.hour
    df['dayofweek'] = df['date_time'].dt.dayofweek
    df['month'] = df['date_time'].dt.month
    df['day'] = df['date_time'].dt.day
    df['quarter'] = df['date_time'].dt.quarter
    df['year'] = df['date_time'].dt.year
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
    df['is_holiday'] = (((df['month'] == 1) & (df['day'] == 1)) | ((df['month'] == 12) & (df['day'] == 25))).astype(int)

    # Cyclical features
    df['sin_hour'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['cos_hour'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['sin_month'] = np.sin(2 * np.pi * df['month'] / 12)
    df['cos_month'] = np.cos(2 * np.pi * df['month'] / 12)
    df['sin_dayofweek'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
    df['cos_dayofweek'] = np.cos(2 * np.pi * df['dayofweek'] / 7)

    # Time of day patterns
    df['morning_peak'] = ((df['hour'] >= 7) & (df['hour'] <= 9)).astype(int)
    df['mid_day'] = ((df['hour'] >= 10) & (df['hour'] <= 14)).astype(int)
    df['evening_peak'] = ((df['hour'] >= 17) & (df['hour'] <= 20)).astype(int)
    df['night'] = ((df['hour'] >= 22) | (df['hour'] <= 5)).astype(int)


    # # Features selection based on correlation analysis
    X = df.drop(columns=['date_time', 'USEP', 'LOAD', 'predicted_load'])

    # # Define target variable (USEP to be predicted)
    y = df['USEP']

    # # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # # Standardize the data (feature scaling)
    scaler_x = StandardScaler()
    X_train_scaled = scaler_x.fit_transform(X_train)
    X_test_scaled = scaler_x.transform(X_test)

    # Optuna objective function for hyperparameter tuning
    def objective(trial):
        # Define hyperparameter space for LightGBM
        param = {
            'objective': 'regression',
            'metric': 'mae',
            'boosting_type': 'gbdt',
            'num_leaves': trial.suggest_int('num_leaves', 20, 100),
            'learning_rate': trial.suggest_loguniform('learning_rate', 0.001, 0.1),
            'feature_fraction': trial.suggest_uniform('feature_fraction', 0.6, 1.0),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
            'subsample': trial.suggest_uniform('subsample', 0.6, 1.0),
        }

        # Create the dataset for LightGBM
        train_data = lgb.Dataset(X_train_scaled, label=y_train)
        valid_data = lgb.Dataset(X_test_scaled, label=y_test, reference=train_data)

        # Train the model
        lgb_model = lgb.train(param,
                            train_data,
                            num_boost_round=1000,
                            valid_sets=[train_data, valid_data])

        # Get predictions and evaluate
        y_pred = lgb_model.predict(X_test_scaled, num_iteration=lgb_model.best_iteration)
        mae = mean_absolute_error(y_test, y_pred)
        return mae  # Return MAE to be minimized

    # Create an Optuna study to optimize the objective function
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=50)  # Number of trials can be adjusted

    # Print the best hyperparameters found
    print("Best hyperparameters:", study.best_params)

    # Train the model using the best hyperparameters
    best_params = study.best_params
    best_lgb_model = lgb.train(best_params,
                            lgb.Dataset(X_train_scaled, label=y_train),
                            num_boost_round=1000,
                            valid_sets=[lgb.Dataset(X_test_scaled, label=y_test)])

    # Save the trained model and the scaler
    best_lgb_model.save_model('best_lgb_model.pkl')
    joblib.dump(scaler_x, 'scaler_x_best_lgb.pkl')

    # Predictions
    y_pred_best = best_lgb_model.predict(X_test_scaled, num_iteration=best_lgb_model.best_iteration)

    # Calculate the metrics
    mae_best = mean_absolute_error(y_test, y_pred_best)
    rmse_best = np.sqrt(mean_squared_error(y_test, y_pred_best))
    r2_best = r2_score(y_test, y_pred_best)
    mape_best = np.mean(np.abs((y_test - y_pred_best) / y_test)) * 100

    # Print results
    print(f"Test MAE: {mae_best}")
    print(f"Test RMSE: {rmse_best}")
    print(f"Test R2 Score: {r2_best}")
    print(f"Test MAPE: {mape_best:.2f}%")

    # Plot predictions vs actual values on some values of test set
    plt.figure(figsize=(12, 6))
    plt.plot(y_test.values[:100], label='True USEP', color='blue')
    plt.plot(y_pred_best[:100], label='Predicted USEP', color='red')
    plt.title('USEP Prediction vs True Values (First 100 Samples)')
    plt.xlabel('Sample Index')
    plt.ylabel('USEP Price')
    plt.legend()
    plt.grid(True)
    plt.savefig('USEP_prediction_best_plot.png')


def usep_model(input_file):
    # Load the dataset
    df = pd.read_csv(input_file)
    df['date_time'] = pd.to_datetime(df['date_time'])

    # Feature Engineering
    df['hour'] = df['date_time'].dt.hour
    df['dayofweek'] = df['date_time'].dt.dayofweek
    df['month'] = df['date_time'].dt.month
    df['day'] = df['date_time'].dt.day
    df['quarter'] = df['date_time'].dt.quarter
    df['year'] = df['date_time'].dt.year
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
    df['is_holiday'] = (((df['month'] == 1) & (df['day'] == 1)) | ((df['month'] == 12) & (df['day'] == 25))).astype(int)

    # Cyclical features
    df['sin_hour'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['cos_hour'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['sin_month'] = np.sin(2 * np.pi * df['month'] / 12)
    df['cos_month'] = np.cos(2 * np.pi * df['month'] / 12)
    df['sin_dayofweek'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
    df['cos_dayofweek'] = np.cos(2 * np.pi * df['dayofweek'] / 7)

    # Time of day patterns
    df['morning_peak'] = ((df['hour'] >= 7) & (df['hour'] <= 9)).astype(int)
    df['mid_day'] = ((df['hour'] >= 10) & (df['hour'] <= 14)).astype(int)
    df['evening_peak'] = ((df['hour'] >= 17) & (df['hour'] <= 20)).astype(int)
    df['night'] = ((df['hour'] >= 22) | (df['hour'] <= 5)).astype(int)

    best_lgb_model = lgb.Booster(model_file='models/usep/saved_models/best_lgb_model.pkl')
    scaler_x = joblib.load('models/usep/saved_models/scaler_x_best_lgb.pkl')



    X = df.drop(columns=['date_time', 'predicted_load', 'predicted_pv'])

    # predict on all values of the dataset
    X_scaled = scaler_x.transform(X)
    y_pred_all = best_lgb_model.predict(X_scaled, num_iteration=best_lgb_model.best_iteration)


    # Save the predictions to a CSV file with initial columns of the dataset
    df['predicted_USEP'] = y_pred_all
    df['predicted_USEP'] = df['predicted_USEP'].clip(lower=0)  # Ensure no negative predictions

    # drop all the columns that were added for feature engineering
    df.drop(columns=['hour', 'dayofweek', 'month', 'day', 'quarter', 'year', 'is_weekend', 'is_holiday',
                    'sin_hour', 'cos_hour', 'sin_month', 'cos_month', 'sin_dayofweek', 'cos_dayofweek',
                    'morning_peak', 'mid_day', 'evening_peak', 'night'], inplace=True)
    
    return df

