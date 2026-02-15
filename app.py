"""
NeuroMotion AI — Single-command launcher.

Usage:
    python app.py              # Build frontend (if needed) + start server on port 8000
    python app.py --no-build   # Skip frontend build, just start server
    python app.py --port 9000  # Custom port
"""

import subprocess
import sys
import os
import shutil
import argparse

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DIST_DIR = os.path.join(ROOT_DIR, "dist")
SERVER_DIR = os.path.join(ROOT_DIR, "server")


def npm_available() -> bool:
    return shutil.which("npm") is not None


def needs_install() -> bool:
    return not os.path.isdir(os.path.join(ROOT_DIR, "node_modules"))


def needs_build() -> bool:
    if not os.path.isdir(DIST_DIR):
        return True
    # Rebuild if any source file is newer than dist/index.html
    index_html = os.path.join(DIST_DIR, "index.html")
    if not os.path.exists(index_html):
        return True
    dist_mtime = os.path.getmtime(index_html)
    source_dirs = [ROOT_DIR, os.path.join(ROOT_DIR, "components"), os.path.join(ROOT_DIR, "services")]
    for d in source_dirs:
        if not os.path.isdir(d):
            continue
        for f in os.listdir(d):
            if f.endswith((".ts", ".tsx", ".html", ".css")):
                if os.path.getmtime(os.path.join(d, f)) > dist_mtime:
                    return True
    return False


def build_frontend():
    if not npm_available():
        print("ERROR: npm is not installed. Install Node.js from https://nodejs.org")
        print("       Then run: python app.py")
        sys.exit(1)

    if needs_install():
        print("Installing frontend dependencies (first time only)...")
        subprocess.run(["npm", "install"], cwd=ROOT_DIR, check=True, shell=True)

    print("Building frontend...")
    subprocess.run(["npm", "run", "build"], cwd=ROOT_DIR, check=True, shell=True)
    print("Frontend built successfully.")


def main():
    parser = argparse.ArgumentParser(description="NeuroMotion AI Launcher")
    parser.add_argument("--no-build", action="store_true", help="Skip frontend build")
    parser.add_argument("--port", type=int, default=8000, help="Server port (default: 8000)")
    parser.add_argument("--host", default="0.0.0.0", help="Server host (default: 0.0.0.0)")
    args = parser.parse_args()

    # Build frontend if needed
    if not args.no_build:
        if needs_build():
            build_frontend()
        else:
            print("Frontend is up to date.")
    elif not os.path.isdir(DIST_DIR):
        print("WARNING: No built frontend found in dist/. The API will run but there's no UI.")
        print("         Run without --no-build to build the frontend.")

    # Start server
    print(f"\n{'='*50}")
    print(f"  NeuroMotion AI starting on http://localhost:{args.port}")
    print(f"{'='*50}\n")

    import uvicorn
    uvicorn.run(
        "api:app",
        host=args.host,
        port=args.port,
        app_dir=SERVER_DIR,
    )


if __name__ == "__main__":
    main()
