# pipeline/config.py
import os

# Role
ROLE_ARN = os.environ.get("SAGEMAKER_ROLE_ARN")

# If running inside SageMaker Studio, auto-detect the execution role
if ROLE_ARN is None:
    try:
        import sagemaker
        ROLE_ARN = sagemaker.get_execution_role()
    except:
        raise ValueError(
            "SAGEMAKER_ROLE_ARN not set and cannot auto-detect. "
            "Set environment variable or run inside SageMaker Studio."
        )

# Bucket
BUCKET = os.environ.get("BUCKET", "olive-datalake-fall2025")

PIPELINE_NAME = "OliveForecastingPipeline"

# Folders
S3_BASE = f"s3://{BUCKET}/mlops"

S3_PREPROCESS_PREFIX = f"{S3_BASE}/preprocess"
S3_TRAINING_PREFIX = f"{S3_BASE}/training"
S3_EVAL_PREFIX = f"{S3_BASE}/evaluation"
S3_REGISTRY_PREFIX = f"{S3_BASE}/registry"
S3_PIPELINE_PREFIX = f"{S3_BASE}/pipeline"


# Region
REGION = os.environ.get(
    "AWS_REGION",
    os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
)

# Instance defaults
DEFAULT_PROCESSING_INSTANCE_TYPE = "ml.m5.xlarge"
DEFAULT_TRAINING_INSTANCE_TYPE = "ml.m5.xlarge"
DEFAULT_TRAINING_INSTANCE_COUNT = 1

# Evaluation metrics
EVAL_METRIC_NAME = "global.mae"
EVAL_METRIC_THRESHOLD = float(os.environ.get("EVAL_METRIC_THRESHOLD", 0.12))

# Names
PIPELINE_NAME = os.environ.get("PIPELINE_NAME", "OliveForecastPipeline")
MODEL_NAME_PREFIX = os.environ.get("MODEL_NAME_PREFIX", "olive-forecast-model")
ENDPOINT_NAME = os.environ.get("ENDPOINT_NAME", "olive-forecast-endpoint")

# S3 scripts prefix
SCRIPTS_S3_PREFIX = f"s3://{BUCKET}/scripts"
