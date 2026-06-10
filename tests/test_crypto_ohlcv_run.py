"""Crypto-OHLCV runner tests (ADR-0043 §2.3, P3b).

Scenarios are hand-traced so the accounting + next-bar-open semantics are pinned.
The strategies use bare price-threshold conditions (no indicators) so signal
timing is fully controlled by the close series.
"""

import pytest

from pancake_engine.crypto_ohlcv.run import run_crypto_ohlcv
from pancake_engine.crypto_ohlcv.types import CryptoOhlcvSpec, OhlcvDataset


def _spec(
    side="long",
    entry_op="gt",
    entry_rhs=100.0,
    exit_op="lt",
    exit_rhs=100.0,
    *,
    slippage_bps=0.0,
    fee_bps=0.0,
    fraction=1.0,
    capital=1000.0,
) -> CryptoOhlcvSpec:
    return CryptoOhlcvSpec(
        spec_family="crypto-ohlcv-spec",
        spec_version="0.1",
        name="t",
        instrument_id="x",
        strategy={
            "side": side,
            "indicators": [],
            "entry": {
                "op": entry_op,
                "left": {"ref": "price", "field": "close"},
                "right": {"ref": "const", "value": entry_rhs},
            },
            "exit": {
                "op": exit_op,
                "left": {"ref": "price", "field": "close"},
                "right": {"ref": "const", "value": exit_rhs},
            },
            "sizing": {"mode": "fixed_fraction", "value": fraction},
        },  # type: ignore[arg-type]
        costs={"slippage_bps": slippage_bps, "fee_bps": fee_bps},  # type: ignore[arg-type]
        starting_capital=capital,
    )


def _ds(oc: list[tuple[float, float]]) -> OhlcvDataset:
    bars = [
        {"t": i * 60, "open": o, "high": max(o, c), "low": min(o, c), "close": c, "volume": 1.0}
        for i, (o, c) in enumerate(oc)
    ]
    return OhlcvDataset(instrument_id="x", bars=bars)  # type: ignore[arg-type]


def _eq(result) -> list[float]:
    return [p.equity for p in result.equity_curve]


# --- next-bar-open entry + signal exit (long, no costs) -----------------------


def test_long_signal_roundtrip_handtraced():
    # close: 101>100 entry@bar0 → fill bar1 open=100; 95<100 exit@bar2 → fill bar3 open=96
    ds = _ds([(100, 101), (100, 102), (103, 95), (96, 97), (98, 99)])
    r = run_crypto_ohlcv(_spec(), ds)
    assert len(r.trades) == 1
    t = r.trades[0]
    assert t.entry_t == 60 and t.entry_price == 100.0  # filled NEXT bar's open, not bar0
    assert t.exit_t == 180 and t.exit_price == 96.0
    assert t.exit_reason == "signal"
    assert t.bars_held == 2
    assert t.pnl == pytest.approx(-40.0)
    assert _eq(r) == pytest.approx([1000, 1020, 950, 960, 960])
    assert r.meta["ending_capital"] == pytest.approx(960.0)


def test_force_close_at_end_of_data():
    # entry@bar1 (close101) → fill bar2 open=100; never exits → force-close at last close=98
    ds = _ds([(100, 99), (100, 101), (100, 102), (105, 103), (104, 98)])
    r = run_crypto_ohlcv(_spec(exit_rhs=1.0), ds)  # exit close<1 never fires
    assert len(r.trades) == 1
    t = r.trades[0]
    assert t.exit_reason == "end_of_data"
    assert t.exit_price == 98.0 and t.exit_t == 240
    assert t.pnl == pytest.approx(-20.0)
    assert r.meta["ending_capital"] == pytest.approx(980.0)


def test_entry_signal_on_last_bar_never_fills():
    ds = _ds([(100, 99), (100, 99), (100, 105)])  # only last close > 100
    r = run_crypto_ohlcv(_spec(), ds)
    assert r.trades == []
    assert _eq(r) == pytest.approx([1000, 1000, 1000])


def test_flat_strategy_no_trades():
    ds = _ds([(100, 99), (100, 98), (100, 97)])
    r = run_crypto_ohlcv(_spec(entry_rhs=1e9), ds)  # entry never fires
    assert r.trades == []
    assert _eq(r) == pytest.approx([1000, 1000, 1000])
    assert r.meta["ending_capital"] == pytest.approx(1000.0)


# --- short side P&L sign ------------------------------------------------------


def test_short_profits_when_price_falls():
    # short entry@bar0 (99<200) fill bar1 open=100; exit@bar3 (96>95) fill bar4 open=97
    ds = _ds([(100, 99), (100, 90), (88, 85), (84, 96), (97, 98)])
    r = run_crypto_ohlcv(
        _spec(side="short", entry_op="lt", entry_rhs=200, exit_op="gt", exit_rhs=95), ds
    )
    assert len(r.trades) == 1
    t = r.trades[0]
    assert t.side == "short"
    assert t.entry_price == 100.0 and t.exit_price == 97.0
    assert t.pnl == pytest.approx(30.0)  # sold @100, bought back @97
    assert r.meta["ending_capital"] == pytest.approx(1030.0)


# --- costs: slippage direction + fee + cash conservation ---------------------


def test_costs_slippage_direction_and_conservation():
    ds = _ds(
        [(100, 101), (100, 100), (90, 110)]
    )  # entry@bar0 → fill bar1 open=100; force-close @110
    r = run_crypto_ohlcv(_spec(exit_rhs=1.0, slippage_bps=100, fee_bps=50), ds)
    t = r.trades[0]
    assert t.entry_price_quote == 100.0 and t.entry_price == pytest.approx(101.0)  # buy slips UP
    assert t.exit_price_quote == 110.0 and t.exit_price == pytest.approx(108.9)  # sell slips DOWN
    assert t.entry_fee == pytest.approx(5.0)  # notional 1000 * 50bps
    assert t.entry_price > t.entry_price_quote and t.exit_price < t.exit_price_quote
    # round-trip cash conservation: ending == starting + sum(pnl)
    assert r.meta["ending_capital"] == pytest.approx(1000.0 + sum(x.pnl for x in r.trades))


def test_conservation_holds_at_extreme_valid_costs():
    # 90% fee + 50% slippage: extreme but valid (< 100%). The validator bound
    # guarantees fee < notional, so qty stays > 0 and the books still balance.
    ds = _ds([(100, 101), (100, 100), (90, 110)])
    r = run_crypto_ohlcv(_spec(exit_rhs=1.0, fee_bps=9000, slippage_bps=5000), ds)
    assert len(r.trades) == 1
    assert r.meta["ending_capital"] == pytest.approx(1000.0 + sum(x.pnl for x in r.trades))
    assert r.meta["ending_capital"] > 0  # capital never silently destroyed at entry


# --- determinism + hash partition --------------------------------------------


def test_result_hash_deterministic():
    ds = _ds([(100, 101), (100, 102), (103, 95), (96, 97)])
    a = run_crypto_ohlcv(_spec(), ds)
    b = run_crypto_ohlcv(_spec(), ds)
    assert a.result_hash == b.result_hash
    assert len(a.result_hash) == 64
    assert a.engine_mode == "crypto_ohlcv_v1"


def test_result_hash_changes_with_spec():
    ds = _ds([(100, 101), (100, 102), (103, 95), (96, 97)])
    a = run_crypto_ohlcv(_spec(entry_rhs=100.0), ds)
    b = run_crypto_ohlcv(_spec(entry_rhs=100.5), ds)  # different compiled_spec_hash
    assert a.result_hash != b.result_hash


def test_instrument_mismatch_returns_blocked_result():
    # Validation failure → empty result with result_hash="" (consistent with PM
    # run_backtest blocked path; does NOT raise).
    ds = _ds([(100, 101)])
    spec = _spec()
    object.__setattr__(spec, "instrument_id", "other")  # force mismatch
    r = run_crypto_ohlcv(spec, ds)
    assert r.result_hash == ""
    assert not r.validation.ok
    assert any("E_CRYPTO_OHLCV_RUN_INVALID" in e.code for e in r.validation.errors)
    assert r.trades == []
    assert r.equity_curve == []
