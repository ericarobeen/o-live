# O-Live: AI-Powered Olive Oil Market Intelligence

> Advanced AI-powered forecasting and conversational intelligence for olive oil market analysis

🌐 **Website:** [https://ericarobeen.github.io/o-live](https://ericarobeen.github.io/o-live)

📅 **Live Presentation:** Monday, December 9, 2025 | 6:30-8:00 PM PST  
🔗 **Zoom Link:** [https://berkeley.zoom.us/j/94855404992](https://berkeley.zoom.us/j/94855404992)

## Overview

A comprehensive AI platform that combines advanced machine learning forecasting with conversational intelligence, providing both technical depth for analysts and natural language accessibility for business users.

## 🎯 Mission Statement

Help olive oil buyers find the best suppliers by modeling market trends and forecasting prices under different scenarios, making the procurement process more transparent through AI-powered market intelligence.

## 🏗️ Architecture

### Agent-Based Architecture
- **Conversational AI Agent**: AWS Bedrock-powered agent with natural language interface for market queries
- **Monitoring Agent**: Automated data quality monitoring with daily health checks and alerting
- **Retraining Agent**: Automated model retraining pipeline that helps to maintain forecast accuracy using alerting

### Technical Stack
- **ML Pipeline**: AWS SageMaker with Ridge Regression models
- **Conversational AI**: AWS Bedrock with Claude 3 Haiku
- **Data Processing**: AWS Lambda serverless functions
- **Data Storage**: S3 Data Lake with Parquet format
- **Orchestration**: EventBridge for automated workflows
- **Frontend**: Streamlit dashboards

## 🚀 Features

### ML Forecasting Engine
- Weekly price predictions with confidence intervals
- Multi-country market analysis (Italy, Spain, Greece, etc.)
- Seasonal pattern recognition and trend analysis
- Real-time model performance monitoring

### Conversational AI Agent
AWS Bedrock-powered agent that provides natural language access to market intelligence:
- Natural language queries: *"What will Italian olive oil cost next month?"*
- Market comparison: *"Compare Italian and Spanish prices"*
- Price driver analysis: *"Why are prices changing?"*
- Real-time responses with data-backed insights from Lambda function tools

### Automated Agent Operations
- **Monitoring Agent**: Daily data freshness checks and quality validation
- **Retraining Agent**: Automated model updates when performance degrades
- Email alerts for system anomalies
- Event-driven orchestration with AWS EventBridge

## 📊 Live Demo Results

The system provides real market intelligence:

- **Price Forecasts**: Italian extra virgin olive oil predicted at $3.50/liter (next month)
- **Market Analysis**: Spain offers lowest prices at $3.80/liter vs Italy at $4.20/liter
- **Price Drivers**: Ocean freight costs up 15.2%, diesel prices up 8.7%
- **Confidence**: 95% confidence intervals for all predictions

## 🎮 Try It Yourself

### O-Live AI Platform Demo

🌐 **Live Platform:** [http://3.238.252.107:8501/](http://3.238.252.107:8501/)  
🎥 **Demo Video:** [Watch Demo](https://raw.githubusercontent.com/ericarobeen/o-live/main/demo/video1908738861.mp4)

Explore price predictions, market drivers, and natural language queries in one unified interface.

## 🔧 Technical Implementation

### AWS Services Used
- **SageMaker**: ML model training and deployment
- **Bedrock**: Conversational AI with Claude 3 Haiku
- **Lambda**: Serverless function execution
- **S3**: Data lake storage
- **EventBridge**: Event-driven automation
- **IAM**: Security and access management

### Model Performance
- **Algorithm**: Ridge Regression with feature engineering
- **Features**: Lag prices, rolling averages, seasonality, cost pressure
- **Accuracy**: Cross-validated on historical data
- **Update Frequency**: Weekly retraining pipeline

### Data Pipeline
1. **Ingestion**: Historical olive oil price data
2. **Processing**: Feature engineering and validation
3. **Training**: Automated model training with hyperparameter tuning
4. **Deployment**: SageMaker endpoint deployment
5. **Monitoring**: Performance tracking and alerting

## 📊 Data

Multi-source dataset combining retail prices, macroeconomic indicators, freight indices, and tariff schedules, creating a comprehensive week-by-week view of global olive oil markets.

### Data Sources
- **[EU Market Observatory](https://agriculture.ec.europa.eu/data-and-analysis/markets/price-data/price-monitoring-sector/olive-oil_en)**: Weekly retail/wholesale olive oil prices by grade and country
- **[FRED (Federal Reserve)](https://fred.stlouisfed.org/series/DEXUSEU)**: Daily EUR/USD exchange rates for price alignment
- **[Freightos Baltic Index (FBX)](https://fbx.freightos.com)**: Weekly ocean freight benchmarks
- **[EIA](https://www.eia.gov/petroleum/gasdiesel/)**: Weekly U.S. diesel prices for transportation costs
- **[USITC](https://hts.usitc.gov/)**: U.S. duty rates and tariff schedules
- **[BLS PPI](https://www.bls.gov/ppi/)**: Monthly packaging material costs (glass, plastic, steel)

### Automated Data Pipeline
Fully automated, quality-controlled data pipeline that:
- **Format Standardization**: Unified data schemas across sources
- **Temporal Alignment**: Weekly timestamp synchronization
- **Range Validation**: Automated quality checks and anomaly detection
- **ML-Ready Output**: Clean economic panel for forecasting and supply-chain insights

## 🤖 Model

Advanced machine learning architecture deployed on AWS SageMaker for scalable, production-ready forecasting.

### Ridge Regression
Regularized linear model that handles multicollinearity in market drivers while maintaining interpretability:
- Handles correlated features effectively
- Provides confidence intervals for predictions
- Interpretable coefficients for market analysis

### Feature Engineering
Sophisticated preprocessing pipeline capturing temporal patterns and market dynamics:
- Lag features for time series patterns
- Rolling averages for trend smoothing
- Seasonal indicators for cyclical patterns
- Market driver interactions

### AWS SageMaker Deployment
Production-ready ML pipeline with automated training, versioning, and real-time inference endpoints for scalable predictions.

## 👥 Team

Built by students in the [UC Berkeley MIDS](https://www.ischool.berkeley.edu/programs/mids) program:

- **[Brandon Gillenwaters](https://www.linkedin.com/in/brandon-gillenwaters/)** 
- **[Erica Robeen](https://www.linkedin.com/in/erica-robeen/)** 
- **[Mahesh Nidhruva](https://www.linkedin.com/in/mahesh-nidhruva/)** 
- **[Patrick Abousleiman](https://www.linkedin.com/in/patrick-abousleiman/)** 
- **[Rebecca Baugh](https://www.linkedin.com/in/rebecca-baugh/)**

### Special Thanks to Our Project Advisors

**UC Berkeley MIDS Program:**
- [Joyce Shen](https://www.linkedin.com/in/joycejshen/)
- [Korin Reid](https://www.linkedin.com/in/korin-reid-ph-d-b4102a17/)

**Subject Matter Experts:**
- [Audra Kamp](https://www.linkedin.com/in/audrateel/)
- [Thomas Anderson](https://www.linkedin.com/in/thomas-anderson-/)
- [Ravi Siddappa](https://www.linkedin.com/in/rsiddappa/)

## 🎓 Academic Context

This project was developed as part of the UC Berkeley Master of Information and Data Science (MIDS) program, demonstrating:
- Enterprise-grade AWS architecture
- Production-ready AI/ML systems with automated operations
- Real-world application to commodity markets
- Integration of multiple AI/ML services


## 🚀 Future Enhancements

- Expand to additional commodities (wheat, corn, soybeans)
- Real-time news sentiment analysis
- Weather pattern integration
- Mobile application development
- API marketplace for third-party integration

---


*Built using AWS AI/ML services by UC Berkeley MIDS students*
