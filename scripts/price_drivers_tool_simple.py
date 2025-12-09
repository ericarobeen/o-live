"""
Price Drivers Tool - Analyzes factors affecting olive oil prices
Simplified version that provides insights on price drivers
"""

import boto3
import json
from datetime import datetime
from typing import Dict, Any

s3 = boto3.client('s3')


def analyze_price_drivers(country: str = None, time_period: str = '3 months') -> Dict[str, Any]:
    """
    Analyze factors driving olive oil price changes
    
    Args:
        country: Country to analyze (e.g., "Italy", "Spain")
        time_period: Time period for analysis
    
    Returns:
        Dictionary with price driver analysis
    """
    
    # Sample analysis (in production, calculate from actual feature data in S3)
    result = {
        'analysis_date': datetime.now().strftime('%Y-%m-%d'),
        'country': country or 'All Markets',
        'time_period': time_period,
        'price_change': {
            'current_price_usd_per_liter': 4.10,
            'previous_price_usd_per_liter': 3.85,
            'change_percent': 6.5,
            'direction': 'increasing'
        },
        'key_drivers': [
            {
                'factor': 'Ocean Freight Costs',
                'impact': 'High',
                'change_percent': 15.2,
                'direction': 'increasing',
                'explanation': 'Shipping costs have increased due to fuel prices and supply chain constraints'
            },
            {
                'factor': 'USD/EUR Exchange Rate',
                'impact': 'Medium',
                'change_percent': -3.1,
                'direction': 'decreasing',
                'explanation': 'Stronger dollar makes European olive oil relatively cheaper for US buyers'
            },
            {
                'factor': 'Diesel Prices',
                'impact': 'Medium',
                'change_percent': 8.7,
                'direction': 'increasing',
                'explanation': 'Higher diesel costs affect production and transportation'
            },
            {
                'factor': 'Packaging Costs (Glass)',
                'impact': 'Low',
                'change_percent': 2.3,
                'direction': 'stable',
                'explanation': 'Glass bottle costs remain relatively stable'
            }
        ],
        'summary': 'Price increases are primarily driven by rising ocean freight and diesel costs, partially offset by favorable exchange rates.',
        'data_source': 'olive-datalake-fall2025',
        'features_analyzed': ['freight_index', 'usd_eur_rate', 'diesel_price', 'packaging_costs'],
        'last_updated': '2025-11-01'
    }
    
    return result


def lambda_handler(event, context):
    """Lambda handler for Bedrock agent tool"""
    
    try:
        # Extract parameters from Bedrock agent request
        parameters = event.get('parameters', [])
        
        # Parse parameters
        country = None
        time_period = '3 months'
        
        for param in parameters:
            if param['name'] == 'country':
                country = param['value']
            elif param['name'] == 'time_period':
                time_period = param['value']
        
        # Get analysis
        analysis = analyze_price_drivers(country, time_period)
        
        # Return in Bedrock agent format
        return {
            'response': {
                'actionGroup': event['actionGroup'],
                'function': event['function'],
                'functionResponse': {
                    'responseBody': {
                        'TEXT': {
                            'body': json.dumps(analysis)
                        }
                    }
                }
            }
        }
        
    except Exception as e:
        return {
            'response': {
                'actionGroup': event.get('actionGroup', 'price-drivers-tool'),
                'function': event.get('function', 'analyze_price_drivers'),
                'functionResponse': {
                    'responseBody': {
                        'TEXT': {
                            'body': json.dumps({
                                'error': str(e),
                                'message': 'Failed to analyze price drivers'
                            })
                        }
                    }
                }
            }
        }
