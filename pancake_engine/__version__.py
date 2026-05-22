"""Version + engine identity constants for Pancake Engine 0.3.

These constants are written into every result emitted by the runner (PR-1+).
They are also part of `result_hash` — bumping any of them is a deliberate
breaking change to the receipt contract.
"""

__version__ = "0.3.0"
ENGINE = "pancake-engine-py"
ENGINE_VERSION = "0.3.0"
ENGINE_MODE = "event_time_v1"
