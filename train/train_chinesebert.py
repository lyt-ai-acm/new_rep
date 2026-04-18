import os
import subprocess
import sys

if __name__ == "__main__":
    cmd = [sys.executable, os.path.join(os.path.dirname(__file__), "train_baseline.py"), "--backbone", "chinesebert", *sys.argv[1:]]
    raise SystemExit(subprocess.call(cmd))
