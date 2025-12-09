"""
Simplified Monitoring Agent for Olive Oil Price Forecasting
Uses basic statistics without heavy ML dependencies
"""

import boto3
import json
from datetime import datetime, timedelta
from typing import Dict, Any

s3 = boto3.client('s3')
ses = boto3.client('ses')


def calculate_basic_metrics(predictions: list, actuals: list) -> Dict[str, float]:
    """Calculate basic error metrics"""
    if not predictions or not actuals or len(predictions) != len(actuals):
        return {}
    
    errors = [abs(p - a) for p, a in zip(predictions, actuals)]
    mae = sum(errors) / len(errors)
    
    percentage_errors = [abs((p - a) / a) * 100 for p, a in zip(predictions, actuals) if a != 0]
    mape = sum(percentage_errors) / len(percentage_errors) if percentage_errors else 0
    
    return {
        'mae': mae,
        'mape': mape,
        'num_predictions': len(predictions)
    }


def check_data_freshness(bucket: str, prefix: str) -> Dict[str, Any]:
    """Check if data is recent"""
    try:
        response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=1)
        if 'Contents' in response:
            last_modified = response['Contents'][0]['LastModified']
            age_days = (datetime.now(last_modified.tzinfo) - last_modified).days
            return {
                'is_fresh': age_days < 7,
                'age_days': age_days,
                'last_modified': last_modified.isoformat()
            }
    except Exception as e:
        return {'error': str(e)}
    
    return {'is_fresh': False, 'error': 'No data found'}


def send_alert(email: str, subject: str, body: str):
    """Send email alert via SES"""
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


def monitor(bucket: str, email: str) -> Dict[str, Any]:
    """Main monitoring function"""
    results = {
        'timestamp': datetime.now().isoformat(),
        'checks': {},
        'alerts': []
    }
    
    # Check data freshness
    freshness = check_data_freshness(bucket, 'curated/')
    results['checks']['data_freshness'] = freshness
    
    if not freshness.get('is_fresh', False):
            alert_msg = f"Data is stale: {freshness.get('age_days', 'Unknown')} days old. Last updated: {freshness.get('last_modified', 'Unknown')}"
        send_alert(email, 'Olive Oil Data Freshness Alert', alert_msg)
        results['alerts'].append('Data is stale')
    
    # Check for recent predictions (placeholder - would need actual prediction data)
    results['checks']['predictions'] = {
        'status': 'No prediction data available yet',
        'note': 'Deploy models to enable prediction monitoring'
    }
    
    # Summary
    results['summary'] = {
        'total_checks': len(results['checks']),
        'alerts_sent': len(results['alerts']),
        'status': 'WARNING' if results['alerts'] else 'OK'
    }
    
    return results
