"""Start Isaac Sim the same way in every script (PhysX engine, optional headless)."""

from __future__ import annotations

from typing import Any


def start_app(headless: bool, width: int = 1280, height: int = 720) -> Any:
    from isaacsim import SimulationApp

    return SimulationApp(
        {
            "headless": headless,
            "width": width,
            "height": height,
            # Force PhysX: the belt (surface velocity) and the suction gripper need it.
            "extra_args": [
                "--/exts/isaacsim.core.simulation_manager/default_engine=physx",
                "--/exts/isaacsim.physics.newton/auto_switch_on_startup=false",
            ],
        }
    )
