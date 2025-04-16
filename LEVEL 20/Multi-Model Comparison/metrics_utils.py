import json
from typing import Dict, Any

def serialize_metrics(model_type: str, mse: float, rmse: float, r2: float) -> str:
    """Convert metrics to JSON string for reliable serialization."""
    return json.dumps({
        "model_type": model_type,
        "mse": float(mse),
        "rmse": float(rmse),
        "r2": float(r2)
    })

def deserialize_metrics(metrics_str: str) -> Dict[str, Any]:
    """Convert JSON string back to metrics dictionary."""
    return json.loads(metrics_str)