import subprocess
runs = []
# K round 2: base 0.6, production x3 method B, f4, 16 worlds, 8192 tick, chunk 64
base = ["--arm", "forager", "--worlds", "16", "--ticks", "8192", "--ramp-ticks", "0",
        "--founders", "4", "--no-reseed", "--track", "--chunk", "64",
        "--base-metabolism", "0.0012", "--bush-count", "60", "--bush-regen-rate", "0.09",
        "--stock-min", "0.05"]
variants = [
    ("baseline_logitm1", ["--founder-animal-logit", "-1.0"]),
    ("logitm2", ["--founder-animal-logit", "-2.0"]),
    ("mixed_std2", ["--founder-animal-logit", "-1.0", "--founder-logit-std", "2.0"]),
    ("nutregen_x2", ["--founder-animal-logit", "-1.0", "--nut-regen-multiplier", "2.0"]),
]
for name, extra in variants:
    runs.append(base + extra + ["--out", "runs/kcal2/%s.json" % name])
# task-20 pilot: fixed arm, target world base0.6 prod3B, ramp 6000, rescue allowed during ramp
runs.append([
    "--arm", "fixed", "--worlds", "8", "--ticks", "12000", "--ramp-ticks", "6000",
    "--founders", "64", "--track", "--chunk", "64", "--base-metabolism", "0.0012",
    "--bush-count", "60", "--bush-regen-rate", "0.09", "--reseed-until-tick", "6000",
    "--out", "runs/pilot/base06_prod3B_fixed.json",
])
for extra in runs:
    r = subprocess.run([".venv/bin/python", "scripts/diagnostics.py", *extra], capture_output=True, text=True)
    print("RUN", extra[-1], "exit", r.returncode, flush=True)
    print(r.stdout[-3500:], flush=True)
    print(r.stderr[-900:], flush=True)
