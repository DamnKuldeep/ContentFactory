"""
Shared helpers for the per-stage setup scripts (setup_narration/images/bgm/story/compose).

Each setup script creates a DEDICATED venv for its stage, installs that stage's packages into it,
and downloads the stage's models into a repo-local HuggingFace cache
(ContentFactory/models/hf_cache) so the first real run is offline-fast. Per-stage venvs are
deliberate — Fish (torch 2.8 + bitsandbytes), FLUX.2 (diffusers) and ACE-Step have conflicting
deps, so isolating them avoids breakage.
"""

import os
import subprocess
import sys

# Paths
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
CF_ROOT     = os.path.dirname(SCRIPTS_DIR)                 # ContentFactory/
REPO_ROOT   = os.path.dirname(CF_ROOT)                     # project root
HF_CACHE    = os.path.join(CF_ROOT, "models", "hf_cache")  # matches shared.config.HF_HOME

# Base deps every node needs (config/llm/drive/manifest). Mirrors the conda-deps we hit at runtime.
BASE_DEPS = [
    "python-dotenv", "tenacity", "openai", "pydantic", "rich",
    "google-api-python-client", "google-auth", "google-auth-httplib2", "google-auth-oauthlib",
]


def log(msg):
    print(f"[setup] {msg}", flush=True)


def venv_python(venv_dir):
    """Path to the venv's python (Linux/macOS = bin, Windows = Scripts)."""
    b = os.path.join(venv_dir, "bin", "python")
    return b if os.path.exists(os.path.dirname(b)) else os.path.join(venv_dir, "Scripts", "python.exe")


def make_venv(name):
    """Create (if missing) a venv at <repo>/<name>; return its python path."""
    venv_dir = os.path.join(REPO_ROOT, name)
    vpy = venv_python(venv_dir)
    if os.path.exists(vpy):
        log(f"venv exists: {venv_dir}")
    else:
        log(f"creating venv: {venv_dir}")
        subprocess.run([sys.executable, "-m", "venv", venv_dir], check=True)
        vpy = venv_python(venv_dir)
    subprocess.run([vpy, "-m", "pip", "install", "-q", "--upgrade", "pip", "setuptools", "wheel"],
                   check=True)
    return vpy, venv_dir


def pip_install(vpy, packages=None, requirements=None):
    """Install a list of packages and/or a requirements file into the venv."""
    if requirements:
        log(f"pip install -r {requirements}")
        subprocess.run([vpy, "-m", "pip", "install", "-r", requirements], check=True)
    if packages:
        log(f"pip install {' '.join(packages)}")
        subprocess.run([vpy, "-m", "pip", "install", *packages], check=True)


def run_py(vpy, code, extra_env=None):
    """Run a Python snippet with the venv python, HF_HOME set to the repo cache + CF_ROOT on path."""
    env = dict(os.environ)
    env["HF_HOME"] = HF_CACHE
    env["PYTHONPATH"] = CF_ROOT + os.pathsep + env.get("PYTHONPATH", "")
    if extra_env:
        env.update(extra_env)
    os.makedirs(HF_CACHE, exist_ok=True)
    subprocess.run([vpy, "-c", code], check=True, env=env, cwd=CF_ROOT)


def stage_requirements(stage_dirname):
    """Absolute path to a stage's requirements.txt (e.g. 'stage_02_narration')."""
    return os.path.join(CF_ROOT, "stages", stage_dirname, "requirements.txt")


def done(stage, venv_dir):
    log(f"✅ {stage} ready. Activate with:  source {venv_dir}/bin/activate")
    log(f"   Models cached under: {HF_CACHE}")
