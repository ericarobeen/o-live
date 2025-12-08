# O-Live: AI-Powered Olive Oil Market Intelligence

> Advanced AI-powered forecasting and conversational intelligence for olive oil market analysis

🌐 **Website:** [https://ericarobeen.github.io/o-live](https://ericarobeen.github.io/o-live)

📅 **Live Presentation:** Monday, December 9, 2025 | 6:30-8:00 PM PST  
🔗 **Zoom Link:** [https://berkeley.zoom.us/j/94855404992](https://berkeley.zoom.us/j/94855404992)

## Overview

A comprehensive multi-agent AI platform that combines advanced machine learning forecasting with conversational intelligence, providing both technical depth for analysts and natural language accessibility for business users.

## 🎯 Mission Statement

Help olive oil buyers find the best suppliers by modeling market trends and forecasting prices under different scenarios, making the procurement process more transparent through AI-powered market intelligence.

## 🏗️ Architecture

### Multi-Agent System
- **Monitoring Agent**: Automated data quality monitoring with daily health checks
- **Retraining Agent**: Automated model retraining pipeline for forecast accuracy
- **Conversational Agent**: AWS Bedrock-powered natural language interface

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

### Conversational AI Assistant
- Natural language queries: *"What will Italian olive oil cost next month?"*
- Market comparison: *"Compare Italian and Spanish prices"*
- Price driver analysis: *"Why are prices changing?"*
- Real-time responses with data-backed insights

### Automated Operations
- Daily data freshness monitoring
- Automated model retraining when performance degrades
- Email alerts for system anomalies
- Continuous integration/deployment pipeline

## 📊 Live Demo Results

The system provides real market intelligence:

- **Price Forecasts**: Italian extra virgin olive oil predicted at $3.50/liter (next month)
- **Market Analysis**: Spain offers lowest prices at $3.80/liter vs Italy at $4.20/liter
- **Price Drivers**: Ocean freight costs up 15.2%, diesel prices up 8.7%
- **Confidence**: 95% confidence intervals for all predictions

## 🎮 Try It Yourself

### O-Live AI Platform Demo
Interactive demonstration combining ML forecasting with conversational AI for comprehensive market intelligence:
```bash
streamlit run integrated_olive_dashboard.py
```

Explore price predictions, market drivers, and natural language queries in one unified interface.

## 📁 Repository Structure

```
├── index.html                          # Project website
├── README.md                           # This file
├── agents/                             # Multi-agent system
│   ├── monitoring_agent.py            # Data quality monitoring
│   ├── retraining_agent.py            # Model retraining automation
│   └── chart_generator_tool.py        # Visualization utilities
├── bedrock_agent/                      # Conversational AI
│   ├── tools/                         # Lambda function tools
│   │   ├── forecast_tool_simple.py    # Price forecasting
│   │   ├── market_comparison_tool_simple.py  # Market analysis
│   │   └── price_drivers_tool_simple.py      # Price factor analysis
│   └── README.md                      # Bedrock setup guide
├── inference.py                        # SageMaker model inference
├── integrated_olive_dashboard.py       # Complete platform UI
├── brandon_original_dashboard.py       # ML-focused dashboard
└── standalone_agent_demo.py           # Conversational AI demo
```

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

### Multi-Source Dataset
- **Retail Prices**: Weekly olive oil prices across grades and countries
- **Macroeconomic Indicators**: Diesel prices, currency exchange rates
- **Freight Indices**: Ocean shipping costs and logistics data
- **Tariff Schedules**: Trade policy and import/export regulations

### Automated Data Pipeline
Fully automated, quality-controlled data pipeline that:
- Standardizes formats across diverse sources
- Aligns weekly timestamps for temporal consistency
- Validates ranges with automated quality checks
- Produces reliable, ML-ready economic panel for forecasting and supply-chain insights

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

- **[Brandon Gillenwaters](https://www.linkedin.com/in/brandon-gillenwaters/)** - SageMaker ML pipeline, forecasting models, data engineering
- **[Erica Robeen](https://www.linkedin.com/in/erica-robeen/)** - Conversational AI, multi-agent orchestration, Lambda tools
- **[Mahesh Nidhruva](https://www.linkedin.com/in/mahesh-nidhruva/)** - Supply chain analytics, market analysis, procurement optimization
- **[Patrick Abousleiman](https://www.linkedin.com/in/patrick-abousleiman/)** - Data engineering, infrastructure, pipeline development
- **[Rebecca Baugh](https://www.linkedin.com/in/rebecca-baugh/)** - Team coordination, stakeholder engagement, project development

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
- Production-ready multi-agent systems
- Real-world application to commodity markets
- Integration of multiple AI/ML services

## 📈 Business Impact

### Value Proposition
- **Technical Users**: Advanced ML controls and statistical analysis
- **Business Users**: Natural language market intelligence
- **Decision Makers**: Data-driven insights for procurement and trading

### Market Applications
- Commodity trading firms
- Food manufacturing companies
- Agricultural cooperatives
- Supply chain management
- Risk assessment and hedging

## 🚀 Future Enhancements

- Expand to additional commodities (wheat, corn, soybeans)
- Real-time news sentiment analysis
- Weather pattern integration
- Mobile application development
- API marketplace for third-party integration

---

*Built with 🫒 using AWS AI/ML services by UC Berkeley MIDS students*