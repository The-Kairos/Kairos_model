#!/usr/bin/env python3
"""
Install MobileVLM so the vlms_light pipeline can use it.
Run from project root: python test/vlms_light/install_mobilevlm.py
"""
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def main():
    print("Installing MobileVLM...")
    # Try pip install from git first
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "git+https://github.com/Meituan-AutoML/MobileVLM.git"],
            check=True,
            capture_output=False,
        )
        print("MobileVLM installed via pip.")
        return 0
    except subprocess.CalledProcessError as e:
        print(f"pip install failed: {e}")
        print("Trying clone + PYTHONPATH...")

    # Fallback: clone and add to PYTHONPATH
    clone_dir = PROJECT_ROOT / "MobileVLM"
    if not clone_dir.exists():
        subprocess.run(
            ["git", "clone", "https://github.com/Meituan-AutoML/MobileVLM.git", str(clone_dir)],
            check=True,
            capture_output=False,
        )
    if (clone_dir / "mobilevlm").exists():
        print(f"\nMobileVLM cloned to {clone_dir}")
        # Patch: low_cpu_mem_usage=True + device_map triggers meta device error with newer transformers.
        # Change to False so model loads normally (then test_mobilevlm moves to GPU).
        mobilevlm_py = clone_dir / "mobilevlm" / "model" / "mobilevlm.py"
        if mobilevlm_py.exists():
            text = mobilevlm_py.read_text(encoding="utf-8")
            if "low_cpu_mem_usage=True" in text and "low_cpu_mem_usage=False" not in text.split("from_pretrained")[1].split("\n")[0]:
                text = text.replace(
                    "model = MobileLlamaForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)",
                    "model = MobileLlamaForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=False, **kwargs)",
                )
                mobilevlm_py.write_text(text, encoding="utf-8")
                print("Patched mobilevlm.py: low_cpu_mem_usage=False (avoids meta device error)")
        # Install minimal deps (timm, einops, etc.) - full requirements has flash-attn which fails to build.
        minimal_deps = ["timm", "einops", "einops-exts", "shortuuid"]
        print("Installing minimal MobileVLM dependencies (timm, einops, etc.)...")
        r = subprocess.run(
            [sys.executable, "-m", "pip", "install", *minimal_deps],
            capture_output=False,
        )
        if r.returncode != 0:
            print("Warning: some minimal deps failed. Try: pip install timm einops einops-exts shortuuid")
        print("\nAdd to PYTHONPATH before running:")
        print(f"  export PYTHONPATH={clone_dir}:$PYTHONPATH   # Linux/Mac")
        print(f"  set PYTHONPATH={clone_dir};%PYTHONPATH%   # Windows")
        print("\nOr run the pipeline from this directory with PYTHONPATH set.")
        return 0
    else:
        print("Clone succeeded but mobilevlm module not found. Check repo structure.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
