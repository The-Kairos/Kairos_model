#!/bin/bash
# Install all dependencies for test_light_vlms
# Run from project root: bash test_light_vlms/install_deps.sh

cd "$(dirname "$0")/.."
echo "Installing all dependencies for test_light_vlms..."
pip install -r test_light_vlms/requirements_full.txt
echo ""
echo "Done. Run: python test_light_vlms/main_test.py"
