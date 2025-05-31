"""
Advanced monitoring and analytics capabilities.
"""

from typing import Any, Dict, List, Optional
import time
from datetime import datetime
import psutil
import GPUtil
from prometheus_client import Gauge, Counter, start_http_server
import pandas as pd
import numpy as np
from mlflow_assist.utils.helpers import setup_logging

logger = setup_logging()

# Prometheus metrics
MODEL_LATENCY = Gauge("model_inference_latency_seconds", "Model inference latency")
TRAINING_TIME = Counter("model_training_time_seconds", "Total model training time")
MEMORY_USAGE = Gauge("memory_usage_bytes", "Memory usage in bytes")
GPU_UTILIZATION = Gauge("gpu_utilization_percent", "GPU utilization percentage")
API_REQUESTS = Counter("api_requests_total", "Total API requests")

class PerformanceMonitor:
    """Monitor system and model performance."""
    
    def __init__(self, export_metrics: bool = True):
        self.metrics_history: List[Dict[str, Any]] = []
        self.start_time = time.time()
        
        if export_metrics:
            start_http_server(8000)

    def capture_system_metrics(self) -> Dict[str, float]:
        """Capture system performance metrics."""
        metrics = {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_usage": psutil.disk_usage('/').percent,
            "timestamp": datetime.now().isoformat()
        }
        
        # GPU metrics if available
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                metrics["gpu_utilization"] = gpus[0].load * 100
                metrics["gpu_memory_used"] = gpus[0].memoryUsed
                GPU_UTILIZATION.set(metrics["gpu_utilization"])
        except Exception as e:
            logger.warning(f"Failed to get GPU metrics: {e}")
        
        MEMORY_USAGE.set(psutil.virtual_memory().used)
        self.metrics_history.append(metrics)
        return metrics

    def track_inference(
        self,
        latency: float,
        model_name: str
    ) -> None:
        """Track model inference performance."""
        MODEL_LATENCY.labels(model=model_name).set(latency)
        API_REQUESTS.inc()

    def track_training(
        self,
        duration: float,
        model_name: str,
        metrics: Dict[str, float]
    ) -> None:
        """Track model training performance."""
        TRAINING_TIME.labels(model=model_name).inc(duration)
        
        training_metrics = {
            "duration": duration,
            "model": model_name,
            **metrics,
            "timestamp": datetime.now().isoformat()
        }
        self.metrics_history.append(training_metrics)

    def analyze_performance(
        self,
        timeframe: str = "1h"
    ) -> Dict[str, Any]:
        """
        Analyze performance metrics over time.
        
        Args:
            timeframe: Time window for analysis (e.g., "1h", "1d")
        
        Returns:
            Performance analysis results
        """
        df = pd.DataFrame(self.metrics_history)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        
        # Filter by timeframe
        cutoff = pd.Timestamp.now() - pd.Timedelta(timeframe)
        df = df[df["timestamp"] > cutoff]
        
        analysis = {
            "system_metrics": {
                "cpu_avg": df["cpu_percent"].mean(),
                "cpu_max": df["cpu_percent"].max(),
                "memory_avg": df["memory_percent"].mean(),
                "memory_max": df["memory_percent"].max()
            }
        }
        
        if "gpu_utilization" in df.columns:
            analysis["gpu_metrics"] = {
                "utilization_avg": df["gpu_utilization"].mean(),
                "utilization_max": df["gpu_utilization"].max(),
                "memory_used_avg": df["gpu_memory_used"].mean()
            }
        
        return analysis

class AlertManager:
    """Manage monitoring alerts and notifications."""
    
    def __init__(self):
        self.alerts: List[Dict[str, Any]] = []
        self.thresholds = {
            "cpu_percent": 80,
            "memory_percent": 85,
            "gpu_utilization": 90,
            "api_latency": 1.0  # seconds
        }

    def check_thresholds(
        self,
        metrics: Dict[str, float]
    ) -> List[str]:
        """Check metrics against thresholds."""
        new_alerts = []
        
        for metric, value in metrics.items():
            if metric in self.thresholds and value > self.thresholds[metric]:
                alert = f"Alert: {metric} exceeded threshold ({value:.1f} > {self.thresholds[metric]})"
                new_alerts.append(alert)
                self.alerts.append({
                    "message": alert,
                    "timestamp": datetime.now().isoformat(),
                    "metric": metric,
                    "value": value
                })
        
        return new_alerts

    def get_alerts(
        self,
        timeframe: str = "1d"
    ) -> List[Dict[str, Any]]:
        """Get alerts within timeframe."""
        cutoff = datetime.now() - pd.Timedelta(timeframe)
        return [
            alert for alert in self.alerts
            if pd.Timestamp(alert["timestamp"]) > cutoff
        ]

class ModelAnalytics:
    """Advanced model performance analytics."""
    
    def __init__(self):
        self.performance_logs: List[Dict[str, Any]] = []

    def log_prediction(
        self,
        model_name: str,
        prediction: Any,
        actual: Any,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log model prediction for analysis."""
        log_entry = {
            "model": model_name,
            "prediction": prediction,
            "actual": actual,
            "timestamp": datetime.now().isoformat()
        }
        
        if metadata:
            log_entry.update(metadata)
        
        self.performance_logs.append(log_entry)

    def analyze_model_drift(
        self,
        timeframe: str = "7d"
    ) -> Dict[str, Any]:
        """Analyze model performance drift over time."""
        df = pd.DataFrame(self.performance_logs)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        
        cutoff = pd.Timestamp.now() - pd.Timedelta(timeframe)
        df = df[df["timestamp"] > cutoff]
        
        # Group by day and calculate metrics
        daily_metrics = df.groupby([df["timestamp"].dt.date, "model"]).agg({
            "prediction": ["mean", "std"],
            "actual": ["mean", "std"]
        }).reset_index()
        
        # Calculate drift metrics
        drift_analysis = {}
        for model in df["model"].unique():
            model_data = daily_metrics[daily_metrics["model"] == model]
            drift_analysis[model] = {
                "prediction_drift": float(np.std(model_data["prediction"]["mean"])),
                "actual_drift": float(np.std(model_data["actual"]["mean"])),
                "stability_score": float(
                    1 - abs(
                        np.corrcoef(
                            model_data["prediction"]["mean"],
                            model_data["actual"]["mean"]
                        )[0,1]
                    )
                )
            }
        
        return drift_analysis

