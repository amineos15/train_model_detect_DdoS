import os
import pandas as pd
import numpy as np
import boto3
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)

# Configuration AWS à partir des variables d'environnement
aws_access_key_id = os.environ.get('')
aws_secret_access_key = os.environ.get('')
aws_region = os.environ.get('')
bucket_name = 'attack-predictions'
input_csv = 'network_traffic_data_with_predictions.csv'
output_model_path = 'model.h5'

s3_client = boto3.client(
    's3',
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
    region_name=aws_region
)

def load_data_from_s3(bucket_name, file_key):
    response = s3_client.get_object(Bucket=bucket_name, Key=file_key)
    status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")

    if status == 200:
        print(f"Successfully retrieved file from S3: {file_key}")
        data = pd.read_csv(response.get("Body"))
        return data
    else:
        print(f"Failed to retrieve file from S3: {file_key}")
        return None

def preprocess_data(df):
    columns_to_drop = ["Flow ID", "Src IP", "Src Port", "Dst IP", "Dst Port", "Timestamp"]
    df.drop(columns=columns_to_drop, inplace=True)
    df.drop('Unnamed: 0', inplace=True, axis=1)
    df.drop_duplicates(keep='first', inplace=True)
    pd.set_option('use_inf_as_na', True)
    df.dropna(inplace=True)

    scaler = StandardScaler()
    lb = LabelEncoder()
    X = df.drop('Label', axis=1)
    y = df['Label']
    X_scaled = scaler.fit_transform(X)
    y_encoded = lb.fit_transform(y)
    return X_scaled, y_encoded

def build_model(input_dim):
    model = Sequential()
    model.add(Dense(128, activation='relu', input_dim=input_dim))
    model.add(Dropout(0.1))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(1, activation='sigmoid'))

    opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model

def train_model(X_train, y_train, X_val, y_val):
    model = build_model(X_train.shape[1])

    early_stop_callback = EarlyStopping(monitor='val_loss', patience=5)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.001)

    model.fit(X_train, y_train,
              batch_size=128,
              epochs=30,
              verbose=1,
              validation_data=(X_val, y_val),
              callbacks=[reduce_lr, early_stop_callback])

    return model

def save_model_to_s3(model, bucket_name, model_path):
    model.save(model_path)
    s3_client.upload_file(model_path, bucket_name, os.path.basename(model_path))
    print(f"Model saved to S3: {bucket_name}/{os.path.basename(model_path)}")

if __name__ == "__main__":
    df = load_data_from_s3(bucket_name, input_csv)
    if df is not None:
        X, y = preprocess_data(df)
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

        model = train_model(X_train, y_train, X_val, y_val)
        save_model_to_s3(model, bucket_name, output_model_path)
