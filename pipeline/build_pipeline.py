# pipeline/build_pipeline.py
"""
Builds a SageMaker model-build pipeline:
 - Preprocess (SKLearnProcessor)
 - Train (SKLearn TrainingStep)
 - Evaluate (SKLearnProcessor)
 - Conditional: Register model in Model Registry (RegisterModel) when eval metric <= threshold

All artifacts are written into the configured S3 prefixes defined in pipeline/config.py (Option A).
"""

import os
import boto3
import sagemaker
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.processing import ScriptProcessor
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.inputs import TrainingInput
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.model import Model
from sagemaker.sklearn.model import SKLearnModel

from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.workflow.condition_step import ConditionStep, JsonGet
from sagemaker.workflow.conditions import ConditionLessThanOrEqualTo
from sagemaker.workflow.parameters import (
    ParameterString,
    ParameterInteger,
    ParameterFloat,
)
from sagemaker.workflow.properties import PropertyFile
from sagemaker.model_metrics import MetricsSource, ModelMetrics
from sagemaker.workflow.functions import Join
# from sagemaker import Session # saves files into bucket with no folder structure 


# Import your config values
from pipeline.config import (
    ROLE_ARN,
    BUCKET,
    PIPELINE_NAME,
    DEFAULT_PROCESSING_INSTANCE_TYPE,
    DEFAULT_TRAINING_INSTANCE_TYPE,
    DEFAULT_TRAINING_INSTANCE_COUNT,
    EVAL_METRIC_THRESHOLD,
    S3_PREPROCESS_PREFIX,
    S3_TRAINING_PREFIX,
    S3_EVAL_PREFIX,
    S3_REGISTRY_PREFIX,
    S3_PIPELINE_PREFIX,
)

from pathlib import Path

try:
    # When running as a script
    BASE_DIR = Path(__file__).resolve().parent.parent  # parent of pipeline folder
except NameError:
    # When running in a notebook (no __file__)
    BASE_DIR = Path.cwd().parent  # parent of notebooks folder

# Scripts directory
SCRIPTS_DIR = BASE_DIR / "scripts"
REQUIREMENTS_FILE = SCRIPTS_DIR / "requirements.txt"

sagemaker_session = PipelineSession(default_bucket=BUCKET)

def build_pipeline(
    name = PIPELINE_NAME,
    role: str | None = None,
    bucket: str | None = None,
    processing_instance_type: str = DEFAULT_PROCESSING_INSTANCE_TYPE,
    training_instance_type: str = DEFAULT_TRAINING_INSTANCE_TYPE,
    training_instance_count: int = DEFAULT_TRAINING_INSTANCE_COUNT,
    eval_threshold: float = EVAL_METRIC_THRESHOLD,
    deploy_endpoint: bool = False,
) -> Pipeline:
    """Construct the SageMaker pipeline object."""

    role = role or ROLE_ARN
    bucket = bucket or BUCKET

    # PipelineSession with explicit default bucket forces artifacts into your bucket
    session = sagemaker_session

    # -------------------------
    # Pipeline parameters
    # -------------------------
    processing_instance_param = ParameterString(
        name="ProcessingInstanceType", default_value=processing_instance_type
    )
    training_instance_param = ParameterString(
        name="TrainingInstanceType", default_value=training_instance_type
    )
    training_instance_count_param = ParameterInteger(
        name="TrainingInstanceCount", default_value=training_instance_count
    )
    eval_threshold_param = ParameterFloat(
        name="EvalThreshold", default_value=eval_threshold
    )

    # -------------------------
    # Preprocess (ProcessingStep)
    # -------------------------
    preprocess_processor = SKLearnProcessor(
        framework_version="1.2-1",
        role=ROLE_ARN,
        instance_type="ml.m5.xlarge",
        instance_count=1,
        base_job_name="olive-preprocess",
        sagemaker_session=sagemaker_session,
    )
    
    processing_step = ProcessingStep(
        name="Preprocess",
        processor=preprocess_processor,
        code=str(SCRIPTS_DIR / "preprocess.py"),
        job_arguments=["--bucket", BUCKET],  # pass bucket here
        outputs=[
            ProcessingOutput(
                output_name="processed_data",
                source="/opt/ml/processing/output",
                destination=S3_PREPROCESS_PREFIX,
            )
        ],
    )

    # -------------------------
    # Train (TrainingStep)
    # -------------------------
    sklearn_estimator = SKLearn(
        entry_point="train.py",
        source_dir=str(SCRIPTS_DIR),
        role=role,
        framework_version="1.2-1",
        instance_type=training_instance_param.default_value,
        instance_count=training_instance_count_param.default_value,
        base_job_name="olive-train",
        sagemaker_session=session,
        output_path= S3_TRAINING_PREFIX,
    )

    training_step = TrainingStep(
        name="TrainModels",
        estimator=sklearn_estimator,
        inputs={
            "processed": TrainingInput(
                s3_data=processing_step.properties.ProcessingOutputConfig.Outputs[
                    "processed_data"
                ].S3Output.S3Uri,
                content_type="application/x-parquet",
            )
        },
    )

    # -------------------------
    # Evaluate (ProcessingStep)
    # -------------------------
    eval_processor = SKLearnProcessor(
        framework_version="1.2-1",
        role=role,
        instance_type=processing_instance_param.default_value,
        instance_count=1,
        base_job_name="olive-eval",
        sagemaker_session=session,
    )

    eval_property_file = PropertyFile(
        name="EvalMetrics",
        output_name="eval_output",
        path="metrics.json",  # evaluate.py writes metrics.json to /opt/ml/processing/output
    )

    eval_step = ProcessingStep(
        name="EvaluateModels",
        processor=eval_processor,
        code=os.path.join(SCRIPTS_DIR, "evaluate.py"),
        inputs=[
            ProcessingInput(
                source=processing_step.properties.ProcessingOutputConfig.Outputs[
                    "processed_data"
                ].S3Output.S3Uri,
                destination="/opt/ml/processing/input",
            ),
            ProcessingInput(
                source=training_step.properties.ModelArtifacts.S3ModelArtifacts,
                destination="/opt/ml/processing/model",
            ),
        ],
        outputs=[
            ProcessingOutput(
                output_name="eval_output",
                source="/opt/ml/processing/output",
                destination= S3_EVAL_PREFIX,
            )
        ],
        property_files=[eval_property_file],
        job_arguments=[
            "--processed_path",
            "/opt/ml/processing/input/processed.parquet",
            "--model_dir",
            "/opt/ml/processing/model",
        ],
    )

    # -------------------------
    # Extract eval metric (global.mae) and condition
    # -------------------------
    metric_get = JsonGet(
        step=eval_step,
        property_file=eval_property_file,
        json_path="global.mae",
    )

    cond_lte = ConditionLessThanOrEqualTo(left=metric_get, right=eval_threshold_param)

    # -------------------------
    # Register Model in Model Registry
    # -------------------------
    # Create a Model object referencing training artifacts. RegisterModel will create a model package.
    model_for_register = SKLearnModel(
    model_data=training_step.properties.ModelArtifacts.S3ModelArtifacts,
        role=role,
        entry_point="inference.py",
        source_dir=str(SCRIPTS_DIR),
        framework_version="1.2-1",
        sagemaker_session=sagemaker_session,
    )


    # Wire the evaluation metrics into the model package (so they appear in the registry)
    metrics_s3_uri = Join(
    on="/",
    values=[eval_step.properties.ProcessingOutputConfig.Outputs["eval_output"].S3Output.S3Uri, "metrics.json"]
    )

    model_metrics = ModelMetrics(
        model_statistics=MetricsSource(
            s3_uri=metrics_s3_uri,
            content_type="application/json",
        )
    )

    register_step = RegisterModel(
        name="RegisterModel",
        model=model_for_register,
        content_types=["text/csv"],
        response_types=["application/json"],
        model_package_group_name="OliveForecastModelPackageGroup",
        model_metrics=model_metrics,
        approval_status="Approved",  # set to Approved if you want auto-approve
    )

    # -------------------------
    # Conditional: if eval metric <= threshold -> register model # skipping for testing
    # -------------------------
    #cond_step = ConditionStep(
    #    name="CheckEvalAndRegisterModel",
    #    conditions=[cond_lte],
    #    if_steps=[register_step],
    #    else_steps=[],
    #)

    # -------------------------
    # Assemble Pipeline
    # -------------------------
    pipeline = Pipeline(
        name=PIPELINE_NAME,
        parameters=[
            processing_instance_param,
            training_instance_param,
            training_instance_count_param,
            eval_threshold_param,
        ],
        steps=[processing_step, training_step, eval_step, register_step], #add cond_step if want to set threshold 
        sagemaker_session=session,
    )

    return pipeline


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--deploy",
        action="store_true",
        help="If set, pipeline will attempt to deploy endpoint when eval passes (not implemented here).",
    )
    args = parser.parse_args()

    pipeline = build_pipeline()
    print("Upserting pipeline definition...")
    pipeline.upsert(role_arn=ROLE_ARN)
    print("Starting pipeline execution...")
    execution = pipeline.start()
    print("Started pipeline execution:", execution.arn)
