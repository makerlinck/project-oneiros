import subprocess
base = ["--arm", "forager", "--worlds", "16", "--ticks", "8192", "--ramp-ticks", "0",
        "--founders", "4", "--no-reseed", "--track", "--chunk", "64",
        "--base-metabolism", "0.0012", "--bush-count", "60", "--bush-regen-rate", "0.09",
        "--stock-min", "0.05", "--founder-animal-logit", "-1.0"]
runs = []
for mult in ("3", "4", "6"):
    runs.append(base + ["--nut-regen-multiplier", mult,
                        "--out", "runs/kcal3/nut%s_v3.json" % mult])
runs.append(base + ["--nut-regen-multiplier", "6", "--hunger-threshold", "0.2",
                    "--out", "runs/kcal3/nut6_h02.json"])
for extra in runs:
    r = subprocess.run([".venv/bin/python", "scripts/diagnostics.py", *extra], capture_output=True, text=True)
    print("RUN", extra[-1], "exit", r.returncode, flush=True)
    print(r.stdout[-2500:], flush=True)
    print(r.stderr[-700:], flush=True)
