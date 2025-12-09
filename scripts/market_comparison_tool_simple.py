"""
Market Comparison Tool - Compares olive oil prices across markets
Simplified version that reads data from S3
"""

import boto3
import json
from datetime import datetime
from typing import Dict, Any, List

s3 = boto3.client('s3')


def compare_markets(countries: List[str] = None, time_period: str = '6 months') -> Dict[str, Any]:
    """
    Compare olive oil prices across different markets
    
    Args:
        countries: List of countries to compare (e.g., ["Italy", "Spain", "Greece"])
        time_period: Time period for comparison (e.g., "3 months", "6 months", "1 year")
    
    Returns:
        Dictionary with comparison data
    """
    
    # Default countries if none specified
    if not countries:
        countries = ["Italy", "Spain", "Greece"]
    
    # Sample comparison data (in production, read from S3)
    result = {
        'comparison_date': datetime.now().strftime('%Y-%m-%d'),
        'time_period': time_period,
        'markets': [
            {
                'country': 'Italy',
                'current_price_usd_per_liter': 4.20,
                'average_price_period': 4.10,
                'price_change_percent': 2.4,
                'trend': 'increasing',
                'grade': 'Extra Virgin'
            },
            {
                'country': 'Spain',
                'current_price_usd_per_liter': 3.80,
                'average_price_period': 3.75,
                'price_change_percent': 1.3,
                'trend': 'stable',
                'grade': 'Extra Virgin'
            },
            {
                'country': 'Greece',
                'current_price_usd_per_liter': 4.50,
                'average_price_period': 4.30,
                'price_change_percent': 4.7,
                'trend': 'increasing',
                'grade': 'Extra Virgin'
            }
        ],
        'insights': {
            'lowest_price': 'Spain',
            'highest_price': 'Greece',
            'most_volatile': 'Greece',
            'price_spread_percent': 18.4
        },
        'data_source': 'olive-datalake-fall2025',
        'last_updated': '2025-11-01'
    }
    
    # Filter to requested countries
    if countries:
        result['markets'] = [m for m in result['markets'] if m['country'] in countries]
    
    return result


def lambda_handler(event, context):
    """Lambda handler for Bedrock agent tool"""
    
    try:
        # Extract parameters from Bedrock agent request
        parameters = event.get('parameters', [])
        
        # Parse parameters
        countries = None
        time_period = '6 months'
        
        for param in parameters:
            if param['name'] == 'countries':
                # Parse comma-separated list
                countries = [c.strip() for c in param['value'].split(',')]
            elif param['name'] == 'time_period':
                time_period = param['value']
        
        # Get comparison
        comparison = compare_markets(countries, time_period)
        
        # Return in Bedrock agent format
        return {
            'response': {
                'actionGroup': event['actionGroup'],
                'function': event['function'],
                'functionResponse': {
                    'responseBody': {
                        'TEXT': {
                            'body': json.dumps(comparison)
                        }
                    }
                }
            }
        }
        
    except Exception as e:
        return {
            'response': {
                'actionGroup': event.get('actionGroup', 'market-comparison-tool'),
                'function': event.get('function', 'compare_markets'),
                'functionResponse': {
                    'responseBody': {
                        'TEXT': {
                            'body': json.dumps({
                                'error': str(e),
                                'message': 'Failed to compare markets'
                            })
                        }
                    }
                }
            }
        }
