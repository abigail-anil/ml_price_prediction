from django.shortcuts import render, redirect
import boto3
import pandas as pd
import joblib
import os
import io
import time
from sklearn.metrics import mean_absolute_error, r2_score
import json
from django.conf import settings
import matplotlib.pyplot as plt

AWS_REGION = 'us-east-1'
BUCKET_NAME = 'cloudmlbucket'

def predict_from_demo(request):
    context = {}

    if request.method == 'POST':
        model_name = request.POST.get('model')
        s3 = boto3.client('s3', region_name=AWS_REGION)

        # Handle comparison mode FIRST
        if model_name == 'comparison':
            MODELS = ['linear', 'knn', 'randomforest', 'xgboost']
            metrics_all = {}
            preview_rows = {}

            for model in MODELS:
                prefix = f"results/{model}/"

                # Load metrics
                metrics_obj = s3.get_object(Bucket=BUCKET_NAME, Key=prefix + f"{model}_metrics.json")
                metrics = json.loads(metrics_obj['Body'].read().decode('utf-8'))
                metrics_all[model] = metrics

                # Load predictions
                pred_obj = s3.get_object(Bucket=BUCKET_NAME, Key=prefix + f"{model}_predictions.csv")
                df = pd.read_csv(pred_obj['Body'])
                preview_rows[model] = df.head(5).to_dict(orient='records')

            # Build comparison metrics table
            metrics_df = pd.DataFrame(metrics_all).T
            metrics_table = metrics_df.reset_index().rename(columns={'index': 'Model'}).to_dict(orient='records')

            # Chart.js data
            chart_data = {
                "labels": MODELS,
                "mae": [metrics_all[m]['MAE'] for m in MODELS],
                "r2": [metrics_all[m]['R²'] for m in MODELS]
            }

            # Best/worst model summary
            best_mae = metrics_df['MAE'].idxmin()
            worst_mae = metrics_df['MAE'].idxmax()
            best_r2 = metrics_df['R²'].idxmax()
            worst_r2 = metrics_df['R²'].idxmin()

            summary = {
                'ranking_mae': metrics_df['MAE'].sort_values().index.tolist(),
                'metric_used': 'MAE (Mean Absolute Error) — lower is better'
            }


            context.update({
                'metrics_table': metrics_table,
                'summary': summary,
                'comparison_mode': True,
                'selected_model': model_name,
                'preview_rows': preview_rows,
                'chart_data': json.dumps(chart_data)
            })
            return render(request, 'predictor/results.html', context)


        # Model selection and prediction logic
        model_map = {
            'xgboost': 'model/xgboost_model.pkl',
            'randomforest': 'model/rf_model.pkl',
            'knn': 'model/knn_model.pkl',
            'linear': 'model/linear_model.pkl'
        }

        model_key = model_map.get(model_name, model_map['randomforest'])
        model_stream = io.BytesIO()
        s3.download_fileobj(BUCKET_NAME, model_key, model_stream)
        model_stream.seek(0)  # rewind to the beginning
        model = joblib.load(model_stream)

        # Download demo data
        demo_path = os.path.join('predictor', 'demo_data.csv')
        with open(demo_path, 'wb') as f:
            s3.download_fileobj(BUCKET_NAME, 'dataset/demo_data.csv', f)
        demo_df = pd.read_csv(demo_path)

        s3_prefix = f"results/{model_name}/"
        X_demo = demo_df.drop(columns=['price'])
        y_actual = demo_df['price']

        # Predict
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

        prediction_table = [
            {'Predicted Price': round(y_pred[i], 2), 'Actual Price': round(y_actual.iloc[i], 2)}
            for i in range(len(y_pred))
        ]

        # Save predictions to S3
        prediction_df = pd.DataFrame(prediction_table)
        csv_buf = io.StringIO()
        prediction_df.to_csv(csv_buf, index=False)
        s3.put_object(Bucket=BUCKET_NAME, Key=s3_prefix + f"{model_name}_predictions.csv", Body=csv_buf.getvalue())

        # Metrics
        mae = mean_absolute_error(y_actual, y_pred)
        r2 = r2_score(y_actual, y_pred)
        metrics = {
            'MAE': round(mae, 2),
            'R²': round(r2, 2),
            'Execution Time (s)': round(execution_time, 4),
            'Throughput (preds/sec)': round(throughput, 2),
            'Latency (s)': round(latency, 6)
        }

        s3.put_object(Bucket=BUCKET_NAME, Key=s3_prefix + f"{model_name}_metrics.json", Body=json.dumps(metrics))

        # Chart.js data
        graph_data = json.dumps({
            "predicted": list(map(float, y_pred)),
            "actual": list(map(float, y_actual))
        })

        # Error Distribution
        errors = y_actual - y_pred
        plt.figure()
        plt.hist(errors, bins=20, color='red', alpha=0.7)
        plt.title('Error Distribution')
        plt.xlabel('Error (Actual - Predicted)')
        plt.ylabel('Frequency')
        plt.tight_layout()
        error_dist_filename = 'error_dist.png'
        error_dist_path = os.path.join(settings.MEDIA_ROOT, error_dist_filename)
        os.makedirs(os.path.dirname(error_dist_path), exist_ok=True)
        plt.savefig(error_dist_path)
        plt.close()

        # Residual Plot
        residuals = y_actual - y_pred
        plt.figure()
        plt.scatter(range(len(residuals)), residuals)
        plt.axhline(0, color='red', linestyle='--')
        plt.title('Residual Plot')
        plt.xlabel('Sample Index')
        plt.ylabel('Residual (Actual - Predicted)')
        plt.tight_layout()
        plot_filename = "residual_plot.png"
        plot_path = os.path.join(settings.MEDIA_ROOT, plot_filename)
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        plt.savefig(plot_path)
        plt.close()

        with open(plot_path, 'rb') as f:
            s3.upload_fileobj(f, BUCKET_NAME, s3_prefix + f"{model_name}_residual_plot.png")
        with open(error_dist_path, 'rb') as f:
            s3.upload_fileobj(f, BUCKET_NAME, s3_prefix + f"{model_name}_error_dist_plot.png")

        # Top Errors Table
        error_df = demo_df.copy()
        error_df['Predicted'] = y_pred.round(2)
        error_df['Actual'] = y_actual.round(2)
        error_df['Absolute Error'] = abs(y_actual - y_pred).round(2)
        top_errors = error_df[['Predicted', 'Actual', 'Absolute Error']].sort_values('Absolute Error', ascending=False).head(5)

        context.update({
            'table': prediction_table,
            'metrics': metrics,
            'graph_data': graph_data,
            'top_errors': top_errors.to_dict(orient='records'),
            'residual_plot': settings.MEDIA_URL + plot_filename,
            'error_dist_plot': settings.MEDIA_URL + error_dist_filename,
            'selected_model': model_name
        })

    return render(request, 'predictor/results.html', context)
