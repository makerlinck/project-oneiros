import subprocess
runs = []
# clustering null-model snapshot on the frozen target world
runs.append([
    "--arm", "forager", "--worlds", "16", "--ticks", "8192", "--ramp-ticks", "0",
    "--founders", "4", "--no-reseed", "--track", "--chunk", "64", "--snapshots",
    "--base-metabolism", "0.0012", "--bush-count", "60", "--bush-regen-rate", "0.09",
    "--nut-regen-multiplier", "3", "--stock-min", "0.05", "--founder-animal-logit", "-1.0",
    "--out", "runs/kcal_snapshot/nut3_f4_snapshots.json",
])
# slow ramp control: fixed vs random-genome control
for arm in ("fixed", "control"):
    runs.append([
        "--arm", arm, "--worlds", "8", "--ticks", "70000", "--ramp-ticks", "60000",
        "--founders", "64", "--track", "--chunk", "64", "--base-metabolism", "0.0012",
        "--bush-count", "60", "--bush-regen-rate", "0.09", "--nut-regen-multiplier", "3",
        "--reseed-until-tick", "60000", "--ramp-metabolism", "0.1",
        "--out", "runs/slowramp/%s.json" % arm,
    ])
for extra in runs:
    r = subprocess.run([".venv/bin/python", "scripts/diagnostics.py", *extra], capture_output=True, text=True)
    print("RUN", extra[-1], "exit", r.returncode, flush=True)
    print(r.stdout[-3000:], flush=True)
    print(r.stderr[-900:], flush=True)
