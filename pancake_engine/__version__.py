"""Version + engine identity constants for Pancake Engine 0.3.

These constants are written into every result emitted by the runner (PR-1+).
They are also part of `result_hash` — bumping any of them is a deliberate
breaking change to the receipt contract.
"""

__version__ = "0.3.1"
ENGINE = "pancake-engine-py"
ENGINE_VERSION = "0.3.0"
ENGINE_MODE = "event_time_v1"

# Verification grade the engine self-identifies with (rule 159 / ADR-0035 §2.2).
# This is a CONTRACT TOKEN tied to the engine *generation* (the 0.3 trust-layer
# per ADR-0031), NOT the package ``__version__``. It must stay in the
# dispatcher's ``PaperVerificationGrade`` union
# (``engine-0.3-canonical | ts-shim-degraded``); changing it is a deliberate
# cross-repo contract change. The release version (``__version__``) advances
# additively (0.3.1 added ``tick()``) without moving the grade, because the
# receipt/fill identity (``ENGINE_VERSION``, part of ``result_hash``) is pinned.
ENGINE_VERIFICATION_GRADE = "engine-0.3-canonical"
