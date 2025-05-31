# Contributing to MLFlow-Assist

First off, thank you for considering contributing to MLFlow-Assist! It's people like you that make MLFlow-Assist such a great tool.

## Code of Conduct

This project and everyone participating in it is governed by our Code of Conduct. By participating, you are expected to uphold this code.

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check the issue list as you might find out that you don't need to create one. When you are creating a bug report, please include as many details as possible:

* Use a clear and descriptive title
* Describe the exact steps which reproduce the problem
* Provide specific examples to demonstrate the steps
* Describe the behavior you observed after following the steps
* Explain which behavior you expected to see instead and why
* Include any error messages

### Suggesting Enhancements

If you have a suggestion for the project, we'd love to hear it. Enhancement suggestions are tracked as GitHub issues.

### Pull Requests

* Fill in the required template
* Do not include issue numbers in the PR title
* Follow the Python styleguides
* Include tests for new features
* Document new code based on the Documentation Styleguide

## Development Process

1. Fork the repo
2. Create a new branch
3. Make your changes
4. Write or adapt tests as needed
5. Update documentation as needed
6. Submit a pull request

## Development Setup

```bash
# Clone your fork
git clone https://github.com/your-username/mlflow-assist.git

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e ".[dev]"

# Run tests
pytest
```

## Style Guidelines

* Use [Black](https://github.com/psf/black) for code formatting
* Use [isort](https://pycqa.github.io/isort/) for import sorting
* Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guide

## Questions?

Feel free to open an issue with your question or contact the maintainers directly.

Thank you for your contributions! 🎉

