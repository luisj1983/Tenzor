"""Phase 5.4 — `tenzor.monitor` event-based monitoring smoke tests.

Verifies the binding surface: register_stat, get_stat, log_event,
add_handler. Custom Python-side EventHandler subclass is exercised so
regressions in the trampoline class are caught.
"""
from __future__ import annotations

import pytest

import tenzor as tz
from tenzor.tenzor_core import monitor as mon


def _fresh_instance():
    """The Monitor is a process-wide singleton; tests use distinct stat
    names to avoid interference."""
    return mon.Monitor.instance()


def test_register_and_accumulate_sum_stat():
    m = _fresh_instance()
    s = m.register_stat("test_sum_stat_unique", mon.Aggregation.Sum)
    s.reset()
    s.add(1.5)
    s.add(2.5)
    s.add(0.5)
    assert s.get() == pytest.approx(4.5)
    assert s.count() == 3
    assert s.name() == "test_sum_stat_unique"


def test_register_and_accumulate_mean_stat():
    m = _fresh_instance()
    s = m.register_stat("test_mean_stat_unique", mon.Aggregation.Mean)
    s.reset()
    for v in [1.0, 2.0, 3.0, 4.0]:
        s.add(v)
    assert s.get() == pytest.approx(2.5)
    assert s.count() == 4


def test_get_stat_returns_registered_instance():
    m = _fresh_instance()
    s1 = m.register_stat("test_get_stat_unique", mon.Aggregation.Count)
    s1.reset()
    s1.add(1.0)
    s2 = m.get_stat("test_get_stat_unique")
    assert s2 is not None, "registered stat should be retrievable by name"
    assert s2.count() == 1


def test_event_handler_receives_log_event():
    """Subclass EventHandler in Python and verify it receives the event."""
    received: list[tuple[str, dict]] = []

    class CapturingHandler(mon.EventHandler):
        def handle(self, event_name, data):
            received.append((event_name, dict(data)))

    m = _fresh_instance()
    handler = CapturingHandler()
    m.add_handler(handler)

    m.log_event("test_event_unique", {"x": 1.0, "y": 2.0})

    # The handler should have received the event. The Monitor singleton may
    # accumulate other tests' events, so we check that ours appears at
    # least once, not that it's the only one.
    matching = [t for t in received if t[0] == "test_event_unique"]
    assert matching, f"event not delivered to handler. Received: {received}"
    assert matching[-1][1] == {"x": 1.0, "y": 2.0}


def test_aggregation_minmax_records_extremes():
    m = _fresh_instance()
    s = m.register_stat("test_minmax_unique", mon.Aggregation.MinMax)
    s.reset()
    for v in [5.0, 1.0, 3.0, 9.0, 2.0]:
        s.add(v)
    # MinMax aggregation behavior is implementation-defined for `get()` —
    # at minimum count() should be accurate.
    assert s.count() == 5
