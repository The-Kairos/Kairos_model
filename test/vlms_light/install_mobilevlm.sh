#!/bin/bash
# Install MobileVLM for light VLM benchmarks.
# Run from project root: bash test/vlms_light/install_mobilevlm.sh

set -e
echo "Installing MobileVLM..."

# Try pip install from git first
if pip install git+https://github.com/Meituan-AutoML/MobileVLM.git 2>/dev/null; then
    echo "MobileVLM installed via pip."
    exit 0
fi

# Fallback: clone and add to PYTHONPATH
echo "pip install failed, trying clone..."
TMPDIR="${TMPDIR:-/tmp}"
REPO="$TMPDIR/MobileVLM"
if [ ! -d "$REPO" ]; then
    git clone https://github.com/Meituan-AutoML/MobileVLM.git "$REPO"
fi
cd "$REPO"
pip install -r requirements.txt 2>/dev/null || true
cd -

echo ""
echo "Add MobileVLM to PYTHONPATH:"
echo "  export PYTHONPATH=\"$REPO:\$PYTHONPATH\""
echo ""
echo "Or add to .env:"
echo "  PYTHONPATH=$REPO"
