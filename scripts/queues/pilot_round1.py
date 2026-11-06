import subprocess
runs = []
for arm in ("fixed", "control"):
    runs.append([
        "--arm", arm, "--worlds", "8", "--ticks", "25000", "--ramp-ticks", "15000",
        "--founders", "64", "--track", "--chunk", "64", "--base-metabolism", "0.0012",
        "--bush-count", "60", "--bush-regen-rate", "0.09", "--reseed-until-tick", "15000",
        "--ramp-metabolism", "0.1",
        "--out", "runs/pilot2/%s.json" % arm,
    ])
for extra in runs:
    r = subprocess.run([".venv/bin/python", "scripts/diagnostics.py", *extra], capture_output=True, text=True)
    print("RUN", extra[-1], "exit", r.returncode, flush=True)
    print(r.stdout[-3000:], flush=True)
    print(r.stderr[-900:], flush=True)
