import os
import psutil
from pathlib import Path
from dotenv import load_dotenv

def is_low_mem() -> bool:
    """
    Check if the system is in low memory mode.
    Returns True if RAM < 24GB or KAIROS_LOW_MEM=TRUE.
    """
    low_mem_env = os.getenv("KAIROS_LOW_MEM", "AUTO").upper()
    if low_mem_env == "AUTO":
        total_ram = psutil.virtual_memory().total / (1024**3)
        return total_ram < 24.0
    return low_mem_env == "TRUE"

def find_project_root() -> Path:
    """
    Starting from the current file's directory, walk upwards until
    a project marker (.env, .env.local, or .git) is found. 
    Returns the Path to the project root.
    """
    markers = [".env", ".env.local", ".git"]
    current_path = Path(__file__).resolve().parent
    # Start searching from current_path and go up
    for parent in [current_path] + list(current_path.parents):
        for marker in markers:
            if (parent / marker).exists():
                return parent
    
    # Fallback to current working directory if not found in parent tree
    return Path(os.getcwd())

def load_kairos_env(override: bool = True):
    """
    Find the project root and load any available environment files (.env, .env.local).
    Ensures correct precedence (local overrides global) and portability.
    """
    root = find_project_root()
    
    # Potential environment file locations
    # Precedence: we load them in order of priority (higher priority last with override=True)
    env_paths = [
        root / "services" / "kairos-python" / "model" / ".env",
        root / ".env",
        root / ".env.local"
    ]
    
    for env_path in env_paths:
        if env_path.exists():
            load_dotenv(env_path, override=override)
    
    return root
