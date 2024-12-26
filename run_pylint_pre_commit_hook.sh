#!/bin/bash
set -x
${UV_PROJECT_ENVIRONMENT:-.venv}/bin/pylint -E $@
