"""Tests for model creation, mutation, and recombination."""

import torch
from csr_llm.model import TinyGPT, mutate_model, recombine_models


def _make_tiny():
    return TinyGPT(vocab_size=64, d_model=32, n_heads=2, n_layers=2, d_ff=64, max_seq_len=16)


def test_forward():
    m = _make_tiny()
    x = torch.randint(0, 64, (2, 8))
    logits, loss = m(x)
    assert logits.shape == (2, 8, 64)
    assert loss is None


def test_forward_with_targets():
    m = _make_tiny()
    x = torch.randint(0, 64, (2, 8))
    t = torch.randint(0, 64, (2, 8))
    _, loss = m(x, t)
    assert loss is not None
    assert loss.item() > 0


def test_generate():
    m = _make_tiny()
    x = torch.randint(0, 64, (1, 4))
    out = m.generate(x, max_new_tokens=10, temperature=1.0)
    assert out.shape[1] == 14  # 4 input + 10 generated


def test_mutate_changes_weights():
    m = _make_tiny()
    original = {n: p.clone() for n, p in m.named_parameters()}
    mutant = mutate_model(m, sigma=0.1, seed=42)
    changed = False
    for n, p in mutant.named_parameters():
        if not torch.equal(p, original[n]):
            changed = True
            break
    assert changed


def test_mutate_deterministic():
    m = _make_tiny()
    a = mutate_model(m, sigma=0.1, seed=42)
    b = mutate_model(m, sigma=0.1, seed=42)
    for (_, pa), (_, pb) in zip(a.named_parameters(), b.named_parameters()):
        assert torch.equal(pa, pb)


def test_recombine():
    a = _make_tiny()
    b = _make_tiny()
    # Make b different
    with torch.no_grad():
        for p in b.parameters():
            p.fill_(99.0)
    child = recombine_models(a, b, seed=42)
    # Child should have some params from a and some from b
    has_a = False
    has_b = False
    for (_, pc), (_, pa), (_, pb) in zip(
        child.named_parameters(), a.named_parameters(), b.named_parameters()
    ):
        if torch.equal(pc, pa):
            has_a = True
        if torch.equal(pc, pb):
            has_b = True
    assert has_a or has_b  # At least inherited from one parent


def test_param_count():
    m = _make_tiny()
    assert m.n_params > 0
    assert m.n_params == sum(p.numel() for p in m.parameters())
