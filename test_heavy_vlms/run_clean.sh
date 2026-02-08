#!/bin/bash
# Clear Python cache before running
echo "Cleaning Python cache..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -type f -name "*.pyc" -delete 2>/dev/null
echo "Cache cleared. Running main_test.py..."
python main_test.py
