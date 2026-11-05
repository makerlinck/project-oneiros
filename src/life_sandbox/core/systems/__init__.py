"""Systems, in the fixed execution order of SANDBOX.md §3.3.

Every system is a pure function:

    apply(state: WorldState, cfg: Config) -> tuple[WorldState, EventBuffer]

System file map:
    fields       world/fields.py
    environment  core/systems/environment.py
    sense        core/systems/sensing.py
    think        core/systems/think.py
    learn        core/systems/learn.py   (uses plasticity.py, prediction.py)
    motion       core/systems/motor.py
    collide      world/physics.py
    interact     core/systems/interaction.py
    body         core/systems/metabolism.py
    reproduce    core/systems/reproduction.py
    die          core/systems/death.py
    events       produced by core/step.py itself (buffer concat, no state write)
"""
