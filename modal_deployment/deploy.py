#!/usr/bin/env python3
"""
Deployment script for One-DM handwriting generation on Modal
"""

import subprocess
import sys

def deploy():
    """Deploy the One-DM model to Modal"""
    try:
        print("Deploying One-DM handwriting generation to Modal...")
        
        # Deploy the Modal app
        result = subprocess.run([
            "modal", "deploy", "one_dm_api.py"
        ], check=True, capture_output=True, text=True)
        
        print("Deployment successful!")
        print("Output:", result.stdout)
        
        # Get the endpoint URL
        print("\nTo get your endpoint URL, run:")
        print("modal app list")
        print("modal app show one-dm-handwriting")
        
    except subprocess.CalledProcessError as e:
        print(f"Deployment failed: {e}")
        print("Error output:", e.stderr)
        sys.exit(1)
    except FileNotFoundError:
        print("Modal CLI not found. Please install with: pip install modal")
        sys.exit(1)

if __name__ == "__main__":
    deploy()