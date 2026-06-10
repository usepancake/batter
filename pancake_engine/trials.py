"""TrialHistory — externally-supplied search-trial ledger for DSR computation.

The platform (pancake 0.10.0+) supplies the caller's TRUE search history so
deflated_sharpe_ratio is computed against the real trial count, not the
sensitivity sweep's synthetic grid. This module is intentionally small: it is a
validated value-object that travels as an execution argument to run_backtest.

Design note:
  - TrialHistory is an EXECUTION argument (like with_inference), NOT part of
    config / config_hash. The result_hash is byte-identical whether or not
    trial_history is supplied. This is the contract: the caller's search history
    is not a property of the run itself, only of the significance assessment.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = ["TrialHistory"]


@dataclass(frozen=True)
class TrialHistory:
    """Caller-supplied search-trial ledger for Deflated Sharpe Ratio.

    Args:
        trial_sharpes: Tuple of per-trial Sharpes from the caller's search
            history (the ledger of all strategies tried before this one, on
            the same dataset). Must have at least 1 entry; all values must be
            finite floats.
        annualized: Whether the supplied Sharpes are annualized (True) or
            per-period (False). Passed directly to deflated_sharpe_ratio's
            ``sharpes_annualized`` flag.
        source: Free-text provenance tag, e.g.
            "platform-ledger:search_session=abc123". Non-empty.

    The current run's own Sharpe is ALWAYS appended to trial_sharpes when DSR
    is computed (the run is itself a trial; omitting it would undercount the
    search and overstate significance). The output block records
    ``own_sharpe_included: true`` to document this.
    """

    trial_sharpes: tuple[float, ...]
    annualized: bool
    source: str

    def __post_init__(self) -> None:
        if len(self.trial_sharpes) < 1:
            raise ValueError(
                "E_TRIAL_HISTORY_EMPTY: trial_sharpes must have at least 1 entry"
            )
        bad = [s for s in self.trial_sharpes if not math.isfinite(s)]
        if bad:
            raise ValueError(
                f"E_TRIAL_HISTORY_NON_FINITE: trial_sharpes contains non-finite values: {bad[:5]}"
            )
        if not self.source or not self.source.strip():
            raise ValueError(
                "E_TRIAL_HISTORY_EMPTY_SOURCE: source must be a non-empty string"
            )
