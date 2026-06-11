import torch

from src.model import BatchedToyModels


def make_model(widths=(2, 5), n=8, seed=0):
    return BatchedToyModels(list(widths), n=n, generator=torch.Generator().manual_seed(seed))


def test_forward_shapes():
    model = make_model()
    x = torch.rand(2, 16, 8)
    assert model(x).shape == (2, 16, 8)
    # A shared (B, n) batch broadcasts over the model dimension.
    assert model(torch.rand(16, 8)).shape == (2, 16, 8)


def test_masked_rows_have_no_effect():
    model = make_model(widths=(2, 5))
    x = torch.rand(2, 32, 8)
    out = model(x)
    with torch.no_grad():
        model.W[0, 2:, :] = 1e6  # garbage in rows beyond model 0's width
    assert torch.allclose(model(x), out)


def test_masked_rows_get_zero_gradient():
    model = make_model(widths=(2, 5))
    model(torch.rand(2, 32, 8)).sum().backward()
    assert torch.all(model.W.grad[0, 2:, :] == 0)
    assert torch.any(model.W.grad[0, :2, :] != 0)


def test_matches_manual_computation():
    model = make_model(widths=(3,), n=4)
    x = torch.rand(1, 5, 4)
    W = model.W_eff[0]
    expected = torch.relu(x[0] @ W.T @ W + model.b[0])
    assert torch.allclose(model(x)[0], expected, atol=1e-6)
