#!/bin/bash
# This script runs the full medical imaging pipeline.
# It ensures that the Python interpreter from the project's virtual environment is used.

echo "Starting the medical imaging pipeline..."
echo "Any arguments passed to this script will be forwarded to the pipeline."

# Run the pipeline using the python executable from the .venv
./.venv/bin/python frontend/utils/pipeline.py "$@"

echo "Pipeline finished."
