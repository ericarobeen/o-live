# O-Live: AI-Powered Olive Oil Market Intelligence

> Advanced AI-powered forecasting and conversational intelligence for olive oil market analysis

🌐 **Website:** [https://ericarobeen.github.io/o-live](https://ericarobeen.github.io/o-live)

## Overview

A comprehensive multi-agent AI platform that combines advanced machine learning forecasting with conversational intelligence, providing both technical depth for analysts and natural language accessibility for business users.

## 🎯 Mission Statement

Deliver the first comprehensive AI-driven solution for olive oil market intelligence, combining advanced machine learning forecasting with conversational AI to enable data-driven decision making in agricultural commodity markets.

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

### ML Forecasting Dashboard
Technical interface for data scientists and analysts:
```bash
streamlit run brandon_original_dashboard.py
```

### Integrated AI Platform
Complete solution with conversational AI:
```bash
streamlit run integrated_olive_dashboard.py
```

### Conversational AI Only
Standalone Bedrock agent demo:
```bash
streamlit run standalone_agent_demo.py
```

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

## 👥 Team

- **Erica**: AI/ML Engineering - Bedrock conversational AI, Lambda tools, multi-agent orchestration
- **Brandon**: ML Engineering - SageMaker ML pipeline, forecasting models, data engineering

## 🎓 Academic Context

This project was developed as part of an AI/ML program, demonstrating:
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

## 📞 Contact

For questions, collaboration, or technical details:
- 📧 Email: [your-email@example.com]
- 💼 LinkedIn: [Your LinkedIn Profile]
- 🐙 GitHub: [Your GitHub Profile]

---

*Built with ❤️ using AWS AI/ML services*