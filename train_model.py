import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib

def train_and_save():
    # Load data
    df = pd.read_csv('heart_diseases.csv')
    
    # Prepare features
    df_processed = df.dropna(subset=['Age', 'Sex', 'BloodPressure', 'Cholesterol', 'MaxHR']).copy()
    
    # Encode categorical variables if needed
    if 'Sex' in df_processed.columns and df_processed['Sex'].dtype == 'object':
        sex_mapping = {'M': 1, 'F': 0}
        df_processed['Sex'] = df_processed['Sex'].replace(sex_mapping)
        
    features = ['Age', 'Sex', 'BloodPressure', 'Cholesterol', 'MaxHR']
    X = df_processed[features].values
    
    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train KMeans
    kmeans = KMeans(n_clusters=5, random_state=56, n_init=10)
    kmeans.fit(X_scaled)
    
    # Save models
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(kmeans, 'kmeans_model.pkl')
    print("Model and scaler saved successfully.")

if __name__ == '__main__':
    train_and_save()
