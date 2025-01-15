import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
    
class SolarDataProcessor:
    def __init__(self, sequence_length=24):
        self.sequence_length = sequence_length
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        
    def load_data(self, filepath):
        """Load and process the solar data"""
        try:
            df = pd.read_csv(filepath)
            df['date_time'] = pd.to_datetime(df['date_time'])
            df.set_index('date_time', inplace=True)
            df = df.sort_index()
            return df
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            raise

    def analyze_data_quality(self, df):
        """Prints out a data quality report"""
        print("\nData Quality Report:")
        print("-" * 50)
        print("\nDataset Shape:", df.shape)
        print("\nFeatures:", df.columns.tolist())
        print("\nMissing Values:")
        print(df.isnull().sum())
        print("\nDuplicate timestamps:", df.index.duplicated().sum())
        print("\nValue Ranges:")
        print(df.describe())

    def clean_data(self, df):
        """Clean the dataset by removing duplicates and handling missing values"""
        initial_rows = len(df)
        
        # Remove duplicates
        df = df[~df.index.duplicated()]
        
        # Handle missing values
        for column in df.columns:
            if df[column].isnull().any():
                if 'power' in column.lower():
                    # Fill missing power values with 0
                    df[column] = df[column].fillna(0)
                else:
                    # Interpolate missing weather data
                    df[column] = df[column].interpolate(method='time').fillna(method='ffill').fillna(method='bfill')
        
        final_rows = len(df)
        if initial_rows != final_rows:
            print(f"Removed {initial_rows - final_rows} duplicate/invalid rows")
        
        return df

    def ensure_continuity(self, df):
        """Ensure that the time series is continuous"""
        initial_rows = len(df)
        
        # Create continuous time range at hourly frequency
        full_time_range = pd.date_range(start=df.index.min(), 
                                      end=df.index.max(), 
                                      freq='H')
        
        # Reindex and fill missing values
        df = df.reindex(full_time_range)
        
        # Interpolate any gaps
        df = df.interpolate(method='time', limit=24)  # Limit interpolation to 24-hour gaps
        
        final_rows = len(df)
        if final_rows > initial_rows:
            print(f"Added {final_rows - initial_rows} timestamps to ensure continuity")
        
        return df

    def prepare_sequences(self, df):
        """Prepare sequences for GAN training"""
        try:
            # Scale the features
            scaled_data = self.scaler.fit_transform(df)
            scaled_df = pd.DataFrame(scaled_data, columns=df.columns, index=df.index)
            
            # Create sequences
            sequences = []
            for i in range(len(scaled_df) - self.sequence_length + 1):
                sequence = scaled_df.iloc[i:i + self.sequence_length].values
                if not np.any(np.isnan(sequence)):  # Only add complete sequences
                    sequences.append(sequence)
            
            sequences = np.array(sequences)
            print(f"Created {len(sequences)} sequences of length {self.sequence_length}")
            
            return sequences, df.columns.tolist()
        
        except Exception as e:
            print(f"Error preparing sequences: {str(e)}")
            raise

    def visualize_data(self, df):
        """Visualize the dataset"""
        try:
            # Create figure with subplots
            fig = plt.figure(figsize=(15, 10))
            
            # 1. Correlation heatmap
            plt.subplot(2, 1, 1)
            sns.heatmap(df.corr(), cmap='coolwarm', center=0, annot=True)
            plt.title('Feature Correlations')
            
            # 2. Time series plot of key variables
            plt.subplot(2, 1, 2)
            if 'power' in df.columns:
                plt.plot(df.index, df['power'], label='Power Output')
            for col in [c for c in df.columns if 'power' not in c.lower()][:3]:  # Plot first 3 non-power features
                plt.plot(df.index, df[col], label=col, alpha=0.5)
            plt.title('Time Series of Key Variables')
            plt.legend()
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"Error in visualization: {str(e)}")

def prepare_data_for_gan(filepath, sequence_length=24):
    try:
        processor = SolarDataProcessor(sequence_length=sequence_length)

        # Load data
        print("Loading data...")
        df = processor.load_data(filepath)

        # Analyze data quality
        print("Analyzing data quality...")
        processor.analyze_data_quality(df)

        # Clean the data
        print("Cleaning data...")
        df = processor.clean_data(df)

        # Ensure time continuity
        print("Ensuring time continuity...")
        df = processor.ensure_continuity(df)

        # Prepare sequences
        print("Preparing sequences...")
        sequences, feature_names = processor.prepare_sequences(df)

        # Visualize the data (optional, can be commented out)
        # print("Generating visualizations...")
        # processor.visualize_data(df)

        return sequences, feature_names, df, processor.scaler

    except Exception as e:
        print(f"Error in data preparation pipeline: {str(e)}")
        raise