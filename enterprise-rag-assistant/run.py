#!/usr/bin/env python3
"""Main entry point to run both backend and frontend"""
import subprocess
import sys
import os
import time
import threading
import signal
from pathlib import Path

def run_backend():
    """Run FastAPI backend"""
    os.environ["PYTHONPATH"] = str(Path.cwd())
    subprocess.run([
        sys.executable, "-m", "uvicorn", "src.api.app:app",
        "--host", "0.0.0.0", "--port", "8000", "--reload"
    ])

def run_frontend():
    """Run Streamlit frontend"""
    time.sleep(2)  # Wait for backend to start
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", "src/ui/app.py",
        "--server.port", "8501", "--server.address", "0.0.0.0"
    ])

if __name__ == "__main__":
    print("Starting Enterprise RAG Assistant...")
    
    #starting backend
    backend_thread = threading.Thread(target=run_backend)
    backend_thread.daemon = True
    backend_thread.start()
    
    #starting frontend
    try:
        run_frontend()
    except KeyboardInterrupt:
        print("\nShutting down...")
        sys.exit(0)