"""
Enterprise deployment automation tools.
"""

from typing import Any, Dict, List, Optional, Union
import os
import subprocess
import docker
import kubernetes
from kubernetes import client, config
import yaml
from pydantic import BaseModel
from mlflow_assist.utils.helpers import setup_logging

logger = setup_logging()

class DeploymentConfig(BaseModel):
    """Deployment configuration."""
    name: str
    model_uri: str
    deployment_type: str = "kubernetes"  # or "docker"
    replicas: int = 1
    resources: Dict[str, str] = {
        "cpu": "1",
        "memory": "2Gi",
        "gpu": "0"
    }
    scaling: Dict[str, Any] = {
        "min_replicas": 1,
        "max_replicas": 5,
        "target_cpu_utilization": 80
    }
    environment: Dict[str, str] = {}

class DeploymentManager:
    """Manage model deployments across different platforms."""
    
    def __init__(self):
        self.docker_client = docker.from_env()
        try:
            config.load_kube_config()
            self.k8s_apps_v1 = client.AppsV1Api()
            self.k8s_core_v1 = client.CoreV1Api()
        except kubernetes.config.config_exception.ConfigException:
            logger.warning("Kubernetes config not found")

    def deploy(
        self,
        deployment_config: DeploymentConfig
    ) -> Dict[str, Any]:
        """
        Deploy model based on configuration.
        
        Args:
            deployment_config: Deployment configuration
        
        Returns:
            Deployment status and details
        """
        if deployment_config.deployment_type == "kubernetes":
            return self._deploy_kubernetes(deployment_config)
        elif deployment_config.deployment_type == "docker":
            return self._deploy_docker(deployment_config)
        else:
            raise ValueError(f"Unsupported deployment type: {deployment_config.deployment_type}")

    def _deploy_kubernetes(
        self,
        config: DeploymentConfig
    ) -> Dict[str, Any]:
        """Deploy to Kubernetes."""
        # Create deployment
        deployment = client.V1Deployment(
            metadata=client.V1ObjectMeta(name=config.name),
            spec=client.V1DeploymentSpec(
                replicas=config.replicas,
                selector=client.V1LabelSelector(
                    match_labels={"app": config.name}
                ),
                template=client.V1PodTemplateSpec(
                    metadata=client.V1ObjectMeta(
                        labels={"app": config.name}
                    ),
                    spec=client.V1PodSpec(
                        containers=[
                            client.V1Container(
                                name=config.name,
                                image=self._build_model_image(config),
                                ports=[client.V1ContainerPort(container_port=8080)],
                                resources=client.V1ResourceRequirements(
                                    requests=config.resources
                                ),
                                env=[
                                    client.V1EnvVar(name=k, value=v)
                                    for k, v in config.environment.items()
                                ]
                            )
                        ]
                    )
                )
            )
        )
        
        # Create deployment
        self.k8s_apps_v1.create_namespaced_deployment(
            body=deployment,
            namespace="default"
        )
        
        # Create service
        service = client.V1Service(
            metadata=client.V1ObjectMeta(name=config.name),
            spec=client.V1ServiceSpec(
                selector={"app": config.name},
                ports=[client.V1ServicePort(port=80, target_port=8080)]
            )
        )
        
        self.k8s_core_v1.create_namespaced_service(
            body=service,
            namespace="default"
        )
        
        # Create HPA
        hpa = client.V1HorizontalPodAutoscaler(
            metadata=client.V1ObjectMeta(name=config.name),
            spec=client.V1HorizontalPodAutoscalerSpec(
                scale_target_ref=client.V1CrossVersionObjectReference(
                    kind="Deployment",
                    name=config.name,
                    api_version="apps/v1"
                ),
                min_replicas=config.scaling["min_replicas"],
                max_replicas=config.scaling["max_replicas"],
                target_cpu_utilization_percentage=config.scaling["target_cpu_utilization"]
            )
        )
        
        self.k8s_apps_v1.create_namespaced_horizontal_pod_autoscaler(
            body=hpa,
            namespace="default"
        )
        
        return {
            "status": "deployed",
            "service_name": config.name,
            "endpoints": [
                f"http://{config.name}.default.svc.cluster.local"
            ]
        }

    def _deploy_docker(
        self,
        config: DeploymentConfig
    ) -> Dict[str, Any]:
        """Deploy using Docker."""
        image_name = self._build_model_image(config)
        
        # Run container
        container = self.docker_client.containers.run(
            image_name,
            detach=True,
            environment=config.environment,
            ports={'8080/tcp': None},
            resources=self._format_docker_resources(config.resources)
        )
        
        return {
            "status": "deployed",
            "container_id": container.id,
            "ports": container.ports
        }

    def _build_model_image(
        self,
        config: DeploymentConfig
    ) -> str:
        """Build Docker image for model deployment."""
        # Create Dockerfile
        dockerfile = f"""
        FROM python:3.8-slim
        
        RUN pip install mlflow==2.22.0 cloudpickle
        
        COPY {config.model_uri} /model
        
        EXPOSE 8080
        
        CMD ["mlflow", "models", "serve", "-m", "/model", "-p", "8080"]
        """
        
        # Build image
        image_name = f"mlflow-assist/model-{config.name}:latest"
        self.docker_client.images.build(
            path=".",
            dockerfile=dockerfile,
            tag=image_name
        )
        
        return image_name

    def _format_docker_resources(
        self,
        resources: Dict[str, str]
    ) -> Dict[str, Any]:
        """Format resources for Docker."""
        return {
            "cpu_quota": int(float(resources["cpu"]) * 100000),
            "mem_limit": resources["memory"]
        }

class DeploymentMonitor:
    """Monitor deployed models."""
    
    def __init__(self):
        try:
            config.load_kube_config()
            self.k8s_apps_v1 = client.AppsV1Api()
            self.k8s_core_v1 = client.CoreV1Api()
        except kubernetes.config.config_exception.ConfigException:
            logger.warning("Kubernetes config not found")
        
        self.docker_client = docker.from_env()

    def get_deployment_status(
        self,
        name: str,
        deployment_type: str
    ) -> Dict[str, Any]:
        """Get deployment status and metrics."""
        if deployment_type == "kubernetes":
            return self._get_kubernetes_status(name)
        elif deployment_type == "docker":
            return self._get_docker_status(name)
        else:
            raise ValueError(f"Unsupported deployment

