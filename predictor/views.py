from django.shortcuts import render, redirect
import boto3
import pandas as pd
import joblib
import os
import time
from sklearn.metrics import mean_absolute_error, r2_score
import json # Visualization with Chart.js
import matplotlib.pyplot as plt
 
AWS_REGION = 'us-east-1'
BUCKET_NAME = 'cloudmlbucket'
 
def predict_from_demo(request):
    context = {}  # Empty status
 
    if request.method == 'POST':
        model_name = request.POST.get('model')
 

        model_map = {
            'xgboost': 'model/xgboost_model.pkl',
            'randomforest': 'model/rf_model.pkl',
            'knn': 'model/knn_model.pkl',
            'linear': 'model/linear_model.pkl'
        }
 
        model_key = model_map.get(model_name, model_map['randomforest'])
        model_path = os.path.join('predictor', 'temp_model.pkl')
 
        # Download model from S3 bucket
        s3 = boto3.client('s3', region_name=AWS_REGION)
        with open(model_path, 'wb') as f:
            s3.download_fileobj(BUCKET_NAME, model_key, f)
        model = joblib.load(model_path)
 
        # demo_data.csv 
        demo_path = os.path.join('predictor', 'demo_data.csv')
        with open(demo_path, 'wb') as f:
            s3.download_fileobj(BUCKET_NAME, 'dataset/demo_data.csv', f)
        demo_df = pd.read_csv(demo_path)
 
        X_demo = demo_df.drop(columns=['price'])
        y_actual = demo_df['price']
 
        # Prediction
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
 
        # Create Table
        prediction_table = []
        for i in range(len(y_pred)):
            row = {
                'Predicted Price': round(y_pred[i], 2),
                'Actual Price': round(y_actual.iloc[i], 2)
            }
            prediction_table.append(row)
 
        # Performance 
        mae = mean_absolute_error(y_actual, y_pred)
        r2 = r2_score(y_actual, y_pred)
        metrics = {
            'MAE': round(mae, 2),
            'R²': round(r2, 2),
            'Execution Time (s)': round(execution_time, 4),
            'Throughput (preds/sec)': round(throughput, 2),
            'Latency (s)': round(latency, 6)
        }
 
        # JSON data for Chart.js
        graph_data = json.dumps({
            "predicted": list(map(float, y_pred)),
            "actual": list(map(float, y_actual))
        })
        
        # Plot 2: Error Distribution
        errors = y_actual - y_pred
        plt.figure()
        plt.hist(errors, bins=20, color='red', alpha=0.7)
        plt.title('Error Distribution')
        plt.xlabel('Error (Actual - Predicted)')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.tight_layout()
        error_dist_filename = 'error_dist.png'
        error_dist_path = os.path.join('static', error_dist_filename)
        plt.savefig(error_dist_path)
        plt.close()
    
        # Analysis Error 
        error_df = demo_df.copy()
        error_df['Predicted'] = y_pred.round(2)
        error_df['Actual'] = y_actual.round(2)
        error_df['Absolute Error'] = abs(y_actual - y_pred).round(2)
        
        columns_to_show = ['Predicted', 'Actual', 'Absolute Error']
        top_errors = error_df[columns_to_show].sort_values('Absolute Error', ascending=False).head(5)
 

        
        y_actual_np = y_actual.values  # Pandas Series → NumPy array
        y_pred_np = y_pred             # Already a NumPy array from model.predict()
        # Plot1: Residual plot
        residuals = y_actual_np - y_pred_np
        plt.figure()
        plt.scatter(range(len(residuals)), residuals)
        plt.axhline(0, color='red', linestyle='--')
        plt.title('Residual Plot')
        plt.xlabel('Sample Index')
        plt.ylabel('Residual (Actual - Predicted)')
        plt.tight_layout()
        plot_filename = "residual_plot.png"
        plot_path = os.path.join('static', plot_filename)
        plt.savefig(plot_path)
        plt.close()

        context.update({
            'table': prediction_table,
            'metrics': metrics,
            'graph_data': graph_data,
            'top_errors': top_errors.to_dict(orient='records'),
            'residual_plot': plot_filename,
            'error_dist_plot': error_dist_filename,
            'selected_model': model_name
        })
 
    return render(request, 'predictor/results.html', context)