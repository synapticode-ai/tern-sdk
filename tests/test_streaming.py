# Copyright 2026 Gamma Seeds Pte Ltd. All rights reserved.
# Tests for P5 — Streaming Inference API

"""Streaming inference tests: TernToken, stream(), stream_blocking(), on_token callback."""

import pytest


@pytest.fixture(scope="module")
def runtime():
    import tern
    model = tern.convert("distilgpt2", verify=False)
    rt = tern.deploy(model)
    yield rt
    rt.unload()


def test_stream_yields_terntoken(runtime):
    """stream() yields TernToken objects."""
    import tern
    tokens = list(runtime.stream("Hello", max_tokens=5))
    assert len(tokens) >= 1
    assert all(isinstance(t, tern.TernToken) for t in tokens)


def test_stream_final_token(runtime):
    """Last token has is_final=True."""
    tokens = list(runtime.stream("Hello", max_tokens=3))
    assert tokens[-1].is_final is True
    assert all(not t.is_final for t in tokens[:-1])


def test_stream_positions_sequential(runtime):
    """Token positions are 0, 1, 2, ..."""
    tokens = list(runtime.stream("Hello", max_tokens=4))
    positions = [t.position for t in tokens]
    assert positions == list(range(len(tokens)))


def test_stream_blocking_collects(runtime):
    """stream_blocking() returns TernOutput with full text."""
    import tern
    output = runtime.stream_blocking("Hello", max_tokens=5, progress=False)
    assert isinstance(output, tern.TernOutput)
    assert isinstance(output.text, str)
    assert len(output.text) > 0
    assert output.tokens_per_second > 0


def test_on_token_callback(runtime):
    """on_token callback is called once per token."""
    import tern
    calls = []
    list(runtime.stream("Hello", max_tokens=3, on_token=lambda t: calls.append(t)))
    assert len(calls) >= 1
    assert all(isinstance(c, tern.TernToken) for c in calls)
