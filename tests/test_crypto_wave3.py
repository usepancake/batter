"""Wave 3 tests: CryptoOHLCVContract, next_bar_open@1 registry, receipt-grade
crypto results, hand-calculated P&L, determinism, and no-look-ahead.

Every hand-traced value is independently verified below the assertion with
step-by-step comments.  No libm transcendentals; all math is IEEE-exact.
"""

from __future__ import annotations

import pytest

from pancake_engine.contracts import (
    CryptoOHLCVContract,
    DatasetContract,
    contract_for_spec_family,
)
from pancake_engine.crypto_ohlcv.run import ENGINE_MODE, run_crypto_ohlcv
from pancake_engine.crypto_ohlcv.types import CryptoOhlcvSpec, OhlcvDataset
from pancake_engine.fills.registry import EntryFill, resolve


# ---------------------------------------------------------------------------
# Helpers (shared with existing crypto run tests)
# ---------------------------------------------------------------------------


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
        instrument_id="BTC",
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


def _ds(oc: list[tuple[float, float]], instrument_id: str = "BTC") -> OhlcvDataset:
    bars = [
        {
            "t": i * 60,
            "open": o,
            "high": max(o, c),
            "low": min(o, c),
            "close": c,
            "volume": 1.0,
        }
        for i, (o, c) in enumerate(oc)
    ]
    return OhlcvDataset(instrument_id=instrument_id, bars=bars)  # type: ignore[arg-type]


# ===========================================================================
# 1. CryptoOHLCVContract — shape, fields, registry
# ===========================================================================


class TestCryptoOHLCVContract:
    def test_is_dataset_contract(self) -> None:
        assert isinstance(CryptoOHLCVContract, DatasetContract)

    def test_domain(self) -> None:
        assert CryptoOHLCVContract.domain == "crypto_ohlcv"

    def test_time_model(self) -> None:
        assert CryptoOHLCVContract.time_model == "bar_series"

    def test_resolution_semantics_none(self) -> None:
        # crypto has continuous P&L, not binary payout
        assert CryptoOHLCVContract.resolution_semantics is None

    def test_fill_reference_next_bar_open(self) -> None:
        assert CryptoOHLCVContract.fill_reference == "next_bar_open"

    def test_required_roles_names(self) -> None:
        names = {r.name for r in CryptoOHLCVContract.required_roles}
        assert names == {"instrument_id", "bar_period", "bars"}

    def test_contract_is_frozen(self) -> None:
        with pytest.raises((AttributeError, TypeError)):
            CryptoOHLCVContract.domain = "mutated"  # type: ignore[misc]

    def test_registry_resolves_crypto_ohlcv_spec(self) -> None:
        c = contract_for_spec_family("crypto-ohlcv-spec")
        assert c is CryptoOHLCVContract

    def test_registry_pm_still_resolves(self) -> None:
        from pancake_engine.contracts import PredictionMarketContract

        c = contract_for_spec_family("pancake-evidence-spec")
        assert c is PredictionMarketContract


# ===========================================================================
# 2. next_bar_open@1 registry entry — math verification
# ===========================================================================


class TestNextBarOpenV1Registry:
    def test_resolves(self) -> None:
        model = resolve("next_bar_open", 1)
        assert model is not None

    def test_long_entry_matches_run_py_math(self) -> None:
        """Hand-verify next_bar_open@1 long fill against run.py lines 134-140.

        run.py long entry:
          fill = bar.open * (1 + slip)
          qty  = (notional - fee) / fill
          cash -= notional

        With quote=100, notional=1000, slip=10bps, fee=5bps:
          fill = 100 * (1 + 10/10_000) = 100 * 1.001 = 100.1
          fee  = 1000 * (5/10_000) = 0.5
          shares = (1000 - 0.5) / 100.1 = 999.5 / 100.1
        """
        model = resolve("next_bar_open", 1)
        quote = 100.0
        notional = 1000.0
        slip_bps = 10.0
        fee_bps = 5.0

        fill = model.apply_entry(
            quote=quote,
            notional=notional,
            slippage_bps=slip_bps,
            fee_bps=fee_bps,
            side="long",
        )
        BPS = 10_000
        expected_fill_price = 100.0 * (1.0 + 10.0 / BPS)  # 100.1
        expected_fee = 1000.0 * (5.0 / BPS)                # 0.5
        expected_shares = (1000.0 - 0.5) / expected_fill_price

        assert fill.fill_price == pytest.approx(expected_fill_price)
        assert fill.fee == pytest.approx(expected_fee)
        assert fill.shares == pytest.approx(expected_shares)

    def test_short_entry_matches_run_py_math(self) -> None:
        """Hand-verify next_bar_open@1 short fill against run.py lines 141-149.

        run.py short entry:
          fill = bar.open * (1 - slip)   (seller receives less)
          qty  = -(notional / fill)       (negative — short)
          cash += notional - fee

        With quote=100, notional=1000, slip=10bps, fee=5bps:
          fill   = 100 * (1 - 10/10_000) = 100 * 0.999 = 99.9
          fee    = 1000 * (5/10_000) = 0.5
          shares = 1000 / 99.9  (registry returns magnitude; sign is caller's)
        """
        model = resolve("next_bar_open", 1)
        quote = 100.0
        notional = 1000.0
        slip_bps = 10.0
        fee_bps = 5.0

        fill = model.apply_entry(
            quote=quote,
            notional=notional,
            slippage_bps=slip_bps,
            fee_bps=fee_bps,
            side="short",
        )
        BPS = 10_000
        expected_fill_price = 100.0 * (1.0 - 10.0 / BPS)  # 99.9
        expected_fee = 1000.0 * (5.0 / BPS)                # 0.5
        expected_shares = 1000.0 / expected_fill_price

        assert fill.fill_price == pytest.approx(expected_fill_price)
        assert fill.fee == pytest.approx(expected_fee)
        assert fill.shares == pytest.approx(expected_shares)

    def test_default_side_is_long(self) -> None:
        """Omitting side defaults to long — backward-compatible for PM callers."""
        model = resolve("next_bar_open", 1)
        fill_default = model.apply_entry(
            quote=100.0, notional=1000.0, slippage_bps=0.0, fee_bps=0.0
        )
        fill_long = model.apply_entry(
            quote=100.0, notional=1000.0, slippage_bps=0.0, fee_bps=0.0, side="long"
        )
        assert fill_default.fill_price == fill_long.fill_price
        assert fill_default.shares == fill_long.shares


# ===========================================================================
# 3. Hand-calculated P&L — tiny bar series, step-by-step verification
# ===========================================================================


class TestHandCalculatedPnL:
    """A 5-bar LONG series with slippage + fee, fully hand-traced.

    Bars (open, close):
      bar0: (100, 105)  — close > 100 → entry signal fires
      bar1: (102, 98)   — entry fills at open=102; close < 100 → exit signal
      bar2: (99,  97)   — exit fills at open=99; position closed
      bar3: (97,  96)   — flat
      bar4: (96,  95)   — flat

    Costs: slippage=10bps, fee=5bps, fraction=1.0, capital=1000.

    Entry (bar1 open=102):
      slip   = 10/10_000 = 0.001
      fill   = 102 * (1 + 0.001) = 102.102  (long pays more)
      fee    = 1000 * (5/10_000) = 0.5
      shares = (1000 - 0.5) / 102.102 = 999.5 / 102.102
             ≈ 9.79...
      cash   = 1000 - 1000 = 0

    Exit (bar2 open=99):
      gross  = shares * (99 * (1 - 0.001)) = shares * 98.901
      fee    = gross * (5/10_000)
      cash   += gross - fee

    P&L:
      pnl = (+1) * shares * (exit_fill - entry_fill) - entry_fee - exit_fee
          = shares * (98.901 - 102.102) - 0.5 - exit_fee
    """

    def setup_method(self) -> None:
        self.slip = 10.0
        self.fee = 5.0
        self.ds = _ds(
            [(100, 105), (102, 98), (99, 97), (97, 96), (96, 95)]
        )
        self.spec = _spec(slippage_bps=self.slip, fee_bps=self.fee)
        self.r = run_crypto_ohlcv(self.spec, self.ds)

    def test_one_trade(self) -> None:
        assert len(self.r.trades) == 1

    def test_entry_bar_and_price(self) -> None:
        t = self.r.trades[0]
        # Filled at bar1's open (t=60), at 102*(1+10/10000)=102.102
        assert t.entry_t == 60
        assert t.entry_price_quote == pytest.approx(102.0)
        assert t.entry_price == pytest.approx(102.0 * (1 + 10.0 / 10_000))

    def test_exit_bar_and_price(self) -> None:
        t = self.r.trades[0]
        # Signal fires bar1 close; exit fills bar2 open=99*(1-10/10000)=98.901
        assert t.exit_t == 120
        assert t.exit_price_quote == pytest.approx(99.0)
        assert t.exit_price == pytest.approx(99.0 * (1 - 10.0 / 10_000))

    def test_exit_reason_signal(self) -> None:
        assert self.r.trades[0].exit_reason == "signal"

    def test_bars_held(self) -> None:
        # Entry at bar index 1, exit at bar index 2 → bars_held = 1
        assert self.r.trades[0].bars_held == 1

    def test_pnl_hand_calculated(self) -> None:
        """Verify PnL to 6 significant figures.

        entry_fill = 102 * 1.001 = 102.102
        exit_fill  = 99  * 0.999 = 98.901
        entry_fee  = 1000 * 0.0005 = 0.5
        shares     = 999.5 / 102.102
        exit_gross = shares * 98.901
        exit_fee   = exit_gross * 0.0005
        pnl        = shares * (98.901 - 102.102) - 0.5 - exit_fee
        """
        BPS = 10_000
        slip = self.slip / BPS
        fee_rate = self.fee / BPS

        entry_fill = 102.0 * (1.0 + slip)   # 102.102
        exit_fill = 99.0 * (1.0 - slip)     # 98.901
        entry_fee = 1000.0 * fee_rate        # 0.5
        shares = (1000.0 - entry_fee) / entry_fill
        exit_gross = shares * exit_fill
        exit_fee = exit_gross * fee_rate
        expected_pnl = shares * (exit_fill - entry_fill) - entry_fee - exit_fee

        assert self.r.trades[0].pnl == pytest.approx(expected_pnl, rel=1e-9)

    def test_cash_conservation(self) -> None:
        """ending_capital == starting_capital + sum(pnl)"""
        ending = self.r.meta["ending_capital"]
        assert ending == pytest.approx(
            1000.0 + sum(t.pnl for t in self.r.trades), rel=1e-9
        )

    def test_equity_curve_length(self) -> None:
        # One EquityPoint per bar
        assert len(self.r.equity_curve) == 5

    def test_result_hash_non_empty(self) -> None:
        assert len(self.r.result_hash) == 64

    def test_engine_mode(self) -> None:
        assert self.r.engine_mode == "crypto_ohlcv_v1"

    def test_validation_ok(self) -> None:
        assert self.r.validation.ok


# ===========================================================================
# 4. No-look-ahead guarantee
# ===========================================================================


class TestNoLookAhead:
    def test_signal_on_last_bar_mints_no_trade(self) -> None:
        """An entry signal fires only on the LAST bar → no next bar to fill on.

        bars: (100,99), (100,99), (100,105) — only bar2 close > 100.
        There is no bar3, so no fill ever happens.
        """
        ds = _ds([(100, 99), (100, 99), (100, 105)])
        r = run_crypto_ohlcv(_spec(), ds)
        assert r.trades == [], (
            "entry signal on last bar must not mint a trade (no next bar)"
        )
        assert all(p.equity == pytest.approx(1000.0) for p in r.equity_curve)

    def test_signal_on_second_to_last_fills_on_last(self) -> None:
        """Signal on bar N-2 → fills on bar N-1 open (not last bar signal)."""
        # bars: bar0 close99, bar1 close101>100 (signal), bar2 fills at open=102
        ds = _ds([(100, 99), (100, 101), (102, 103)])
        r = run_crypto_ohlcv(_spec(exit_rhs=1.0), ds)  # exit never fires → EOD close
        assert len(r.trades) == 1
        t = r.trades[0]
        assert t.entry_t == 120  # bar2 open (t=2*60=120)
        assert t.exit_reason == "end_of_data"

    def test_three_bar_series_exit_on_last_bar_signal_never_fills(self) -> None:
        """Exit signal fires on last bar → no next bar → close uses EOD path."""
        # entry bar0 close=101 → fills bar1 open=100
        # exit signal bar1 close=99 → would fill bar2 open... but bar2 IS the last bar,
        # and exit fills happen at the START of bar processing (the open):
        # actually the pending="exit" is consumed at bar2's open, so the exit DOES fill.
        # Test the specific case: exit fires on LAST bar (bar2) — no bar3 to fill on.
        # bars: bar0(100,101) bar1(100,102) bar2(103,99)
        # entry fires bar0 (close 101>100) → fills bar1 open=100
        # exit fires bar2 (close 99<100) → sets pending=exit, but loop ends → EOD close
        ds = _ds([(100, 101), (100, 102), (103, 99)])
        r = run_crypto_ohlcv(_spec(), ds)
        assert len(r.trades) == 1
        t = r.trades[0]
        # Exit signal fires bar2 but next iter doesn't exist → force close at bar2 close=99
        assert t.exit_reason == "end_of_data"
        assert t.exit_price == pytest.approx(99.0)


# ===========================================================================
# 5. Determinism — same inputs ×3 → byte-identical hash
# ===========================================================================


class TestDeterminism:
    def test_same_inputs_three_times_same_hash(self) -> None:
        ds = _ds([(100, 101), (100, 102), (103, 95), (96, 97), (98, 99)])
        spec = _spec(slippage_bps=10.0, fee_bps=5.0)
        results = [run_crypto_ohlcv(spec, ds) for _ in range(3)]
        h0 = results[0].result_hash
        assert len(h0) == 64, "result_hash must be a 64-char hex SHA-256"
        assert results[1].result_hash == h0
        assert results[2].result_hash == h0

    def test_different_spec_different_hash(self) -> None:
        ds = _ds([(100, 101), (100, 102), (103, 95), (96, 97)])
        a = run_crypto_ohlcv(_spec(entry_rhs=100.0), ds)
        b = run_crypto_ohlcv(_spec(entry_rhs=100.5), ds)
        assert a.result_hash != b.result_hash

    def test_different_dataset_different_hash(self) -> None:
        spec = _spec()
        ds_a = _ds([(100, 101), (100, 102), (103, 95)])
        ds_b = _ds([(100, 101), (100, 102), (103, 94)])  # last close differs
        a = run_crypto_ohlcv(spec, ds_a)
        b = run_crypto_ohlcv(spec, ds_b)
        assert a.result_hash != b.result_hash


# ===========================================================================
# 6. Validation / blocked path
# ===========================================================================


class TestValidationPath:
    def test_instrument_mismatch_blocked(self) -> None:
        ds = _ds([(100, 101)])
        spec = _spec()
        object.__setattr__(spec, "instrument_id", "ETH")  # force mismatch
        r = run_crypto_ohlcv(spec, ds)
        assert r.result_hash == ""
        assert not r.validation.ok
        assert r.trades == []
        assert r.equity_curve == []
        assert r.compiled_spec_hash == ""
        assert r.dataset_hash == ""

    def test_blocked_meta_has_blocked_flag(self) -> None:
        ds = _ds([(100, 101)])
        spec = _spec()
        object.__setattr__(spec, "instrument_id", "ETH")
        r = run_crypto_ohlcv(spec, ds)
        assert r.meta.get("blocked") is True

    def test_valid_run_validation_ok(self) -> None:
        ds = _ds([(100, 101), (100, 99)])
        r = run_crypto_ohlcv(_spec(), ds)
        assert r.validation.ok


# ===========================================================================
# 7. Hash-policy field list — smoke: to_dict() carries all expected keys
# ===========================================================================


class TestHashPolicyFields:
    """Verify the result envelope contains the hash-policy fields documented in
    crypto_ohlcv/result.py:

    Hashed: engine, engine_version, engine_mode, compiled_spec_hash,
            dataset_hash, metrics, equity_curve, drawdown_curve,
            monthly_returns, trades.
    Not hashed: warnings, validation, meta.
    """

    def setup_method(self) -> None:
        ds = _ds([(100, 101), (100, 99), (99, 98)])
        self.r = run_crypto_ohlcv(_spec(), ds)
        self.d = self.r.to_dict()

    def test_hashes_block_present(self) -> None:
        h = self.d["hashes"]
        assert "compiled_spec_hash" in h
        assert "dataset_hash" in h
        assert "result_hash" in h

    def test_engine_fields(self) -> None:
        assert self.d["engine"] == "batter"
        assert "engine_version" in self.d
        assert self.d["engine_mode"] == ENGINE_MODE

    def test_curves_present(self) -> None:
        assert "equity_curve" in self.d
        assert "drawdown_curve" in self.d
        assert "monthly_returns" in self.d

    def test_validation_in_to_dict(self) -> None:
        assert "validation" in self.d
        assert "ok" in self.d["validation"]

    def test_warnings_in_to_dict(self) -> None:
        assert "warnings" in self.d
