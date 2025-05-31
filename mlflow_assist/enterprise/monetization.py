"""
Enterprise monetization and usage tracking features.
"""

from typing import Dict, List, Optional
import time
from datetime import datetime
import jwt
from pydantic import BaseModel
import requests
from mlflow_assist.utils.helpers import setup_logging

logger = setup_logging()

class UsageMetrics(BaseModel):
    """Usage metrics for tracking."""
    api_calls: int = 0
    model_training_time: float = 0
    inference_calls: int = 0
    data_processed: int = 0
    tokens_used: int = 0

class SubscriptionPlan(BaseModel):
    """Subscription plan details."""
    name: str
    limits: Dict[str, int]
    price: float
    features: List[str]

class EnterpriseManager:
    """Enterprise feature management and monetization."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        subscription_plan: str = "free"
    ):
        self.api_key = api_key
        self.subscription_plan = subscription_plan
        self.metrics = UsageMetrics()
        self.start_time = time.time()
        
        # Define subscription plans
        self.plans = {
            "free": SubscriptionPlan(
                name="Free",
                limits={
                    "api_calls": 100,
                    "model_training_time": 3600,
                    "inference_calls": 1000,
                    "data_processed": 1000000,
                    "tokens_used": 10000
                },
                price=0.0,
                features=["basic_ml", "basic_llm"]
            ),
            "pro": SubscriptionPlan(
                name="Professional",
                limits={
                    "api_calls": 10000,
                    "model_training_time": 86400,
                    "inference_calls": 100000,
                    "data_processed": 100000000,
                    "tokens_used": 1000000
                },
                price=99.99,
                features=["basic_ml", "basic_llm", "advanced_ml", "advanced_llm"]
            ),
            "enterprise": SubscriptionPlan(
                name="Enterprise",
                limits={
                    "api_calls": float("inf"),
                    "model_training_time": float("inf"),
                    "inference_calls": float("inf"),
                    "data_processed": float("inf"),
                    "tokens_used": float("inf")
                },
                price=999.99,
                features=["basic_ml", "basic_llm", "advanced_ml", "advanced_llm", 
                         "distributed", "enterprise"]
            )
        }

    def track_usage(
        self,
        metric: str,
        value: int = 1
    ) -> bool:
        """
        Track usage metrics and check limits.
        
        Args:
            metric: Metric to track
            value: Value to add
        
        Returns:
            True if within limits, False otherwise
        """
        current_value = getattr(self.metrics, metric)
        new_value = current_value + value
        plan_limit = self.plans[self.subscription_plan].limits[metric]
        
        if new_value > plan_limit:
            logger.warning(f"Usage limit exceeded for {metric}")
            return False
        
        setattr(self.metrics, metric, new_value)
        return True

    def generate_license(
        self,
        user_id: str,
        company: Optional[str] = None
    ) -> str:
        """
        Generate license key for enterprise features.
        
        Args:
            user_id: User identifier
            company: Company name
        
        Returns:
            License key
        """
        payload = {
            "user_id": user_id,
            "plan": self.subscription_plan,
            "company": company,
            "issued_at": datetime.utcnow().isoformat(),
            "features": self.plans[self.subscription_plan].features
        }
        
        return jwt.encode(
            payload,
            self.api_key or "default-secret",
            algorithm="HS256"
        )

    def verify_license(self, license_key: str) -> bool:
        """
        Verify license key validity.
        
        Args:
            license_key: License key to verify
        
        Returns:
            True if valid, False otherwise
        """
        try:
            payload = jwt.decode(
                license_key,
                self.api_key or "default-secret",
                algorithms=["HS256"]
            )
            return True
        except jwt.InvalidTokenError:
            return False

    def get_usage_report(self) -> Dict[str, float]:
        """Get current usage metrics and costs."""
        total_time = time.time() - self.start_time
        
        return {
            "metrics": self.metrics.dict(),
            "subscription": {
                "plan": self.subscription_plan,
                "price": self.plans[self.subscription_plan].price,
                "time_used": total_time
            }
        }

    def upgrade_plan(
        self,
        new_plan: str,
        payment_token: Optional[str] = None
    ) -> bool:
        """
        Upgrade subscription plan.
        
        Args:
            new_plan: Target subscription plan
            payment_token: Payment verification token
        
        Returns:
            True if upgrade successful
        """
        if new_plan not in self.plans:
            logger.error(f"Invalid plan: {new_plan}")
            return False
        
        if payment_token:
            # Process payment (implement payment gateway integration)
            self.subscription_plan = new_plan
            logger.info(f"Successfully upgraded to {new_plan}")
            return True
        
        logger.error("Payment token required for plan upgrade")
        return False

class UsageMonitor:
    """Monitor and analyze usage patterns."""
    
    def __init__(self, enterprise_manager: EnterpriseManager):
        self.manager = enterprise_manager
        self.alerts = []

    def check_usage_patterns(self) -> List[str]:
        """Analyze usage patterns and generate recommendations."""
        metrics = self.manager.metrics
        recommendations = []
        
        # Check API usage
        api_usage_ratio = metrics.api_calls / self.manager.plans[
            self.manager.subscription_plan
        ].limits["api_calls"]
        
        if api_usage_ratio > 0.8:
            recommendations.append(
                "API usage is approaching limit. Consider upgrading plan."
            )
        
        # Check model training time
        if metrics.model_training_time > 0.7 * self.manager.plans[
            self.manager.subscription_plan
        ].limits["model_training_time"]:
            recommendations.append(
                "Model training time is high. Consider optimization or plan upgrade."
            )
        
        return recommendations

    def alert_on_threshold(
        self,
        metric: str,
        threshold: float
    ) -> None:
        """Set usage alerts for metrics."""
        current_value = getattr(self.manager.metrics, metric)
        limit = self.manager.plans[
            self.manager.subscription_plan
        ].limits[metric]
        
        if current_value >= threshold * limit:
            alert = f"Alert: {metric} has reached {threshold*100}% of limit"
            self.alerts.append(alert)
            logger.warning(alert)

