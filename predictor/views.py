from django.shortcuts import render
import boto3
import pandas as pd
import joblib
import os
import time
from sklearn.metrics import mean_absolute_error, r2_score

AWS_REGION = 'us-east-1'
BUCKET_NAME = 'cloudmlbucket'

def predict_from_demo(request):
    s3 = boto3.client('s3', region_name=AWS_REGION)

    # Load model
    model_path = os.path.join('predictor', 'rf_model.pkl')
    with open(model_path, 'wb') as f:
        s3.download_fileobj(BUCKET_NAME, 'model/rf_model.pkl', f)
    model = joblib.load(model_path)

    # Load demo data
    demo_path = os.path.join('predictor', 'demo_data.csv')
    with open(demo_path, 'wb') as f:
        s3.download_fileobj(BUCKET_NAME, 'dataset/demo_data.csv', f)
    demo_df = pd.read_csv(demo_path)

    X_demo = demo_df.drop(columns=['price'])
    y_actual = demo_df['price']

    # Prediction metrics
    start_time = time.time()
    y_pred = model.predict(X_demo)
    end_time = time.time()

    execution_time = end_time - start_time
    throughput = len(X_demo) / execution_time

    single_input = X_demo.iloc[[0]]
    start_latency = time.time()
    _ = model.predict(single_input)
    end_latency = time.time()
    latency = end_latency - start_latency

    # Combine results for table
    prediction_table = []
    for i in range(len(y_pred)):
        row = X_demo.iloc[i].to_dict()
        row['Predicted Price'] = round(y_pred[i], 2)
        row['Actual Price'] = round(y_actual.iloc[i], 2)
        prediction_table.append(row)

    mae = mean_absolute_error(y_actual, y_pred)
    r2 = r2_score(y_actual, y_pred)

    metrics = {
        'MAE': round(mae, 2),
        'R²': round(r2, 2),
        'Execution Time (s)': round(execution_time, 4),
        'Throughput (preds/sec)': round(throughput, 2),
        'Latency (s)': round(latency, 6)
    }

    return render(request, 'predictor/results.html', {
        'table': prediction_table,
        'metrics': metrics
    })
