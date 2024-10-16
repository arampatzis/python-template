#!/usr/bin/env sh

# This script is used by pre-commit to run mypy from inside a project's folder.
# Since pre-commit is executed from the git root dir, the script first enters the project's
# and then runs mypy with the local configuration file.



mypy --config-file=pyproject.toml
