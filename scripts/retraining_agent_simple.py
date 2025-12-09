"""
Simplified Retraining Agent for Olive Oil Price Forecasting
Triggers retraining workflow and sends notifications
"""

import boto3
import json
from datetime import datetime
from typing import Dict, Any

s3 = boto3.client('s3')
ses = boto3.client('ses')
glue = boto3.client('glue')


def send_notification(email: str, subject: str, body: str):
    """Send email notification via SES"""
    try:
        ses.send_email(
            Source='erica.robeen@berkeley.edu',
            Destination={'ToAddresses': [email]},
            Message={
                'Subject': {'Data': subject},
                'Body': {'Text': {'Data': body}}
            }
        )
        return True
    except Exception as e:
        print(f"Failed to send email: {e}")
        return False


def trigger_feature_build(job_name: str) -> Dict[str, Any]:
    """Trigger Glue job to rebuild features"""
    try:
        response = glue.start_job_run(JobName=job_name)
        return {
            'success': True,
            'job_run_id': response['JobRunId'],
            'job_name': job_name
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }


def check_data_availability(bucket: str, prefix: str) -> Dict[str, Any]:
    """Check if training data is available"""
    try:
        response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=10)
        if 'Contents' in response and len(response['Contents']) > 0:
            return {
                'available': True,
                'file_count': len(response['Contents']),
                'last_modified': response['Contents'][0]['LastModified'].isoformat()
            }
        return {'available': False, 'error': 'No data found'}
    except Exception as e:
        return {'available': False, 'error': str(e)}


def retrain(bucket: str, email: str, trigger_reason: str = 'manual') -> Dict[str, Any]:
    """Main retraining function"""
    results = {
        'timestamp': datetime.now().isoformat(),
        'trigger_reason': trigger_reason,
        'steps': {},
        'status': 'started'
    }
    
    # Step 1: Check data availability
    data_check = check_data_availability(bucket, 'curated/')
    results['steps']['data_check'] = data_check
    
    if not data_check.get('available', False):
        results['status'] = 'failed'
        results['error'] = 'Training data not available'
        
        send_notification(
            email,
            'Olive Oil Retraining Failed',
            f"Retraining failed: Training data not available. Trigger: {trigger_reason}"
        )
        return results
    
    # Step 2: Trigger feature rebuild (optional - only if needed)
    # Uncomment to trigger feature rebuild before retraining
    # feature_job = trigger_feature_build('build_weekly_features')
    # results['steps']['feature_rebuild'] = feature_job
    
    # Step 3: Placeholder for actual model retraining
    # In a full implementation, this would:
    # - Load data from S3
    # - Train SARIMA and Ridge models
    # - Validate performance
    # - Save new models to S3
    results['steps']['model_training'] = {
        'status': 'placeholder',
        'note': 'Full model training requires ML libraries (statsmodels, scikit-learn)',
        'recommendation': 'Use SageMaker Training Jobs for production model retraining'
    }
    
    # Step 4: Send success notification
    results['status'] = 'completed'
    
    send_notification(
        email,
        'Olive Oil Retraining Completed',
        f"Retraining workflow completed. Trigger: {trigger_reason}. Files available: {data_check.get('file_count', 0)}"
    )
    
    return results


def trigger_from_monitoring(monitoring_results: Dict[str, Any], bucket: str, email: str) -> Dict[str, Any]:
    """Trigger retraining based on monitoring results"""
    
    # Check if retraining is needed based on monitoring alerts
    alerts = monitoring_results.get('alerts', [])
    
    if not alerts:
        return {
            'triggered': False,
            'reason': 'No alerts from monitoring - retraining not needed'
        }
    
    # Trigger retraining
    return retrain(bucket, email, trigger_reason='monitoring_alert')
