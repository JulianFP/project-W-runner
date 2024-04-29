# Welcome to project-W-runner

[![License: AGPLv3](https://img.shields.io/badge/License-agplv3-yellow.svg)](https://opensource.org/license/agpl-v3)
[![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/JulianFP/project-W-runner/ci.yml?branch=main)](https://github.com/JulianFP/project-W-runner/actions/workflows/ci.yml)
[![Documentation Status](https://readthedocs.org/projects/project-W-runner/badge/)](https://project-W-runner.readthedocs.io/)
[![codecov](https://codecov.io/gh/JulianFP/project-W-runner/branch/main/graph/badge.svg)](https://codecov.io/gh/JulianFP/project-W-runner)

## Installation

The Python package `project_W_runner` can be installed from PyPI:

```
python -m pip install project_W_runner
```

## Development installation

If you want to contribute to the development of `project_W_runner`, we recommend
the following editable installation from this repository:

```
git clone git@github.com:JulianFP/project-W-runner.git
cd project-W-runner
python -m pip install --editable .[tests]
```

Having done so, the test suite can be run using `pytest`:

```
python -m pytest
```

## Acknowledgments

This repository was set up using the [SSC Cookiecutter for Python Packages](https://github.com/ssciwr/cookiecutter-python-package).
