from setuptools import setup, find_packages

setup(
    name="mlflow-assist",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "torch>=1.9.0",
        "transformers>=4.20.0",
        "mlflow>=1.20.0",
        "click>=8.0.0",
        "rich>=10.0.0",
        "typer>=0.4.0",
        "pydantic>=1.8.0",
        "python-dotenv>=0.19.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "black>=21.0.0",
            "isort>=5.0.0",
            "flake8>=3.9.0",
            "mypy>=0.910",
        ]
    },
    entry_points={
        "console_scripts": [
            "mlflow-assist=mlflow_assist.cli.cli:app",
        ],
    },
    author="happyvibess",
    description="A comprehensive toolkit for ML and LLM development",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/happyvibess/mlflow-assist",
    project_urls={
        "Documentation": "https://github.com/happyvibess/mlflow-assist/tree/master/docs",
        "Bug Reports": "https://github.com/happyvibess/mlflow-assist/issues",
        "Source Code": "https://github.com/happyvibess/mlflow-assist",
        "Buy me a coffee": "https://www.buymeacoffee.com/happyvibess",
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Software Development :: Build Tools",
        "Framework :: MLflow",
        "Environment :: Console",
        "Natural Language :: English",
    ],
    python_requires=">=3.8",
)

