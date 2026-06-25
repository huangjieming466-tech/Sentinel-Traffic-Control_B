#!/usr/bin/env python3
"""Fix hardcoded DISPLAY=:0 in a Python file, replace with auto-detect."""
import sys, os, glob

path = sys.argv[1] if len(sys.argv) > 1 else "npu_predict_net_fast.py"

with open(path) as f:
    content = f.read()

# The auto-detect snippet
new_code = """for s in sorted(glob.glob("/tmp/.X11-unix/X*")):
    n = int(s.rsplit("X", 1)[-1])
    if n < 100:
        os.environ["DISPLAY"] = f":{n}"
        break
else:
    os.environ["DISPLAY"] = ":0\""""

# Replace hardcoded line
old = 'os.environ["DISPLAY"] = ":0"'
if old in content:
    content = content.replace(old, new_code)
    print(f"[OK] Replaced DISPLAY=:0 with auto-detect in {path}")
else:
    print(f"[SKIP] No hardcoded DISPLAY found in {path}")
    sys.exit(0)

# Ensure glob import exists
if "import glob" not in content:
    content = content.replace("import cv2", "import glob\nimport cv2", 1)
    print("[OK] Added glob import")

with open(path, "w") as f:
    f.write(content)

# Syntax check
import py_compile
try:
    py_compile.compile(path, doraise=True)
    print("[OK] Syntax check passed")
except py_compile.PyCompileError as e:
    print(f"[FAIL] Syntax error: {e}")
    sys.exit(1)
