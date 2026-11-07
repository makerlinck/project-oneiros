import subprocess
runs = []
# (a') fresh individuals under the new live-flesh toxin rule, frozen fractions
for frac in ("0.2", "0.3", "0.4"):
    runs.append([
        "--arm", "fixed", "--worlds", "8", "--ticks", "8192", "--ramp-ticks", "60000",
        "--ramp-fraction", frac, "--founders", "64", "--no-reseed", "--track", "--chunk", "64",
        "--base-metabolism", "0.0012", "--bush-count", "60", "--bush-regen-rate", "0.09",
        "--nut-regen-multiplier", "3", "--out", "runs/diag_a2/f%s.json" % frac,
    ])
# (b') slow ramp with the new rule (attacks kept, live flesh poisonous / non-nutritious)
runs.append([
    "--arm", "fixed", "--worlds", "8", "--ticks", "70000", "--ramp-ticks", "60000",
    "--founders", "64", "--track", "--chunk", "64", "--base-metabolism", "0.0012",
    "--bush-count", "60", "--bush-regen-rate", "0.09", "--nut-regen-multiplier", "3",
    "--reseed-until-tick", "60000", "--ramp-metabolism", "0.1",
    "--out", "runs/diag_b2/fixed_live_toxin1.json",
])
for extra in runs:
    r = subprocess.run([".venv/bin/python", "scripts/diagnostics.py", *extra], capture_output=True, text=True)
    print("RUN", extra[-1], "exit", r.returncode, flush=True)
    print(r.stdout[-2500:], flush=True)
    print(r.stderr[-800:], flush=True)
