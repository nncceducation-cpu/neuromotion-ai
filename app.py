"""
NeuroMotion AI — Single-command launcher.

Usage:
    python app.py                # Streamlit UI (default)
    python app.py --mode ui      # Streamlit UI only
    python app.py --mode api     # FastAPI only (port 8000)
    python app.py --mode both    # FastAPI thread + Streamlit process
    python app.py --port 9000    # Custom FastAPI port
"""

import subprocess
import sys
import os
import threading
import argparse

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
SERVER_DIR = os.path.join(ROOT_DIR, "server")


def start_api(host: str, port: int):
    """Start FastAPI server with uvicorn."""
    import uvicorn
    uvicorn.run(
        "api:app",
        host=host,
        port=port,
        app_dir=SERVER_DIR,
    )


def start_ui():
    """Start Streamlit UI."""
    streamlit_app = os.path.join(ROOT_DIR, "streamlit_app.py")
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", streamlit_app,
         "--server.headless", "true"],
        cwd=ROOT_DIR,
    )


def main():
    parser = argparse.ArgumentParser(description="NeuroMotion AI Launcher")
    parser.add_argument(
        "--mode", choices=["api", "ui", "both"], default="ui",
        help="Run mode: 'api' (FastAPI only), 'ui' (Streamlit only, default), 'both' (API + UI)",
    )
    parser.add_argument("--port", type=int, default=8000, help="FastAPI port (default: 8000)")
    parser.add_argument("--host", default="0.0.0.0", help="FastAPI host (default: 0.0.0.0)")
    args = parser.parse_args()

    print(f"\n{'='*50}")
    print(f"  NeuroMotion AI — mode: {args.mode}")
    print(f"{'='*50}\n")

    if args.mode == "api":
        print(f"Starting FastAPI on http://localhost:{args.port}")
        start_api(args.host, args.port)

    elif args.mode == "ui":
        print("Starting Streamlit UI...")
        start_ui()

    elif args.mode == "both":
        print(f"Starting FastAPI on http://localhost:{args.port} (background thread)")
        api_thread = threading.Thread(
            target=start_api, args=(args.host, args.port), daemon=True,
        )
        api_thread.start()

        print("Starting Streamlit UI...")
        start_ui()


if __name__ == "__main__":
    main()
