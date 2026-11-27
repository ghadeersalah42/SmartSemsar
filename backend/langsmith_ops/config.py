"""
LangSmith LLMOps Configuration for Smart Semsar
"""

import os
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root, so backend/ imports
from backend.ai_config import threshold  # noqa: E402
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[1] / '.env')
except Exception:
    pass

BASE_DIR = Path(__file__).resolve().parent
GOLDEN_DATASETS_DIR = BASE_DIR / "golden_datasets"
PROMPTS_BACKUP_DIR = BASE_DIR / "prompts"
TRACES_DIR = BASE_DIR / "traces"
REPORTS_DIR = BASE_DIR / "reports"

# Ensure directories exist
for d in [GOLDEN_DATASETS_DIR, PROMPTS_BACKUP_DIR, TRACES_DIR, REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# LangSmith Environment Settings
LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2", "true")
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT", "smart-semsar")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY", "")
LANGCHAIN_ENDPOINT = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")

# Dataset Names in LangSmith
DATASET_INTENT = "smart_semsar_intent_golden_set"
DATASET_ORCHESTRATOR = "smart_semsar_orchestrator_golden_set"
DATASET_REQUIREMENTS = "smart_semsar_customer_reqs_golden_set"

# Evaluation Thresholds
MAX_ALLOWED_ACCURACY_DROP = threshold("evaluation.max_allowed_accuracy_drop")  # 2 % - config/ai_config.yaml
MIN_PASSING_ACCURACY = threshold("evaluation.min_passing_accuracy")            # 90 % - config/ai_config.yaml
