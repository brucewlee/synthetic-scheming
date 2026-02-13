"""
Environment file utilities.
"""

import os
from pathlib import Path
from typing import Optional, Dict


def load_env(path: Optional[str] = None) -> Dict[str, str]:
    """
    Load environment variables from a .env file.

    Args:
        path: Path to .env file. If None, searches current directory and parents.

    Returns:
        Dictionary of environment variables that were loaded.
    """
    try:
        from dotenv import load_dotenv, dotenv_values
    except ImportError:
        raise ImportError(
            "python-dotenv is required for load_env(). "
            "Install it with: pip install python-dotenv"
        )

    if path:
        env_file = Path(path)
    else:
        # Search for .env file in current and parent directories
        current = Path.cwd()
        env_file = None
        for parent in [current] + list(current.parents):
            potential = parent / ".env"
            if potential.exists():
                env_file = potential
                break

        if not env_file:
            return {}

    if not env_file.exists():
        raise FileNotFoundError(f"Environment file not found: {env_file}")

    # Load the environment variables
    load_dotenv(env_file, override=True)

    # Return what was loaded for visibility
    return dict(dotenv_values(env_file))
