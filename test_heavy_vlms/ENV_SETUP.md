# Instructions for Setting Up .env on VM

The `.env` file needs to be copied to the VM for ASR to work.

## Option 1: Copy .env file to VM

```bash
# On your local machine, copy the .env file content
# Then on the VM:
cd ~/Kairos_model
nano .env
# Paste the content and save (Ctrl+X, Y, Enter)
```

## Option 2: Set environment variables directly

```bash
# On the VM, add to ~/.bashrc or set before running:
export AZURE_OPENAI_KEY="your-key-here"
export AZURE_OPENAI_ENDPOINT="your-endpoint-here"
export AZURE_OPENAI_DEPLOYMENT="whisper-karios"
export AZURE_OPENAI_API_VERSION="2024-12-01-preview"
```

## Option 3: Skip ASR for now

The code now handles missing ASR credentials gracefully. It will continue with `[ASR unavailable]` in the results.

## Verify .env is loaded

```bash
cd ~/Kairos_model/test_heavy_vlms
python -c "from dotenv import load_dotenv; from pathlib import Path; load_dotenv(Path('../.env')); import os; print('ASR Key:', 'SET' if os.getenv('AZURE_OPENAI_KEY') else 'MISSING')"
```
