"""Small checks of the parts. Run: python tests.py"""
import torch, jepa

G = lambda s=0: torch.Generator().manual_seed(s)


def test_patches_keep_every_point():
    img, _, _ = jepa.pictures(5, G())
    p = jepa.patches(img)
    assert p.shape == (5, 64, 16)
    assert torch.equal(p[0, 9].reshape(4, 4), img[0, 4:8, 4:8])      # patch 9 is row 1, column 1


def test_every_shape_is_different():
    n = len(jepa.SHAPES)
    img = jepa.draw(torch.arange(n), torch.full((n,), 16.0), torch.full((n,), 16.0), torch.full((n,), 6.0), G())
    white = img == 1
    assert all(white[i].sum() > 20 for i in range(n))
    assert all((white[i] != white[j]).sum() > 10 for i in range(n) for j in range(i))


def test_hidden_and_seen_patches_do_not_overlap():
    g = G(3)
    for _ in range(50):
        seen, hidden = jepa.hide(g)
        assert sorted(seen.tolist() + hidden.tolist()) == list(range(64))
        assert 9 <= len(hidden) <= 18


def test_encoder_never_gets_the_hidden_patches():
    enc = jepa.Encoder()
    img, _, _ = jepa.pictures(2, G())
    seen, hidden = jepa.hide(G(1))
    a = enc(jepa.patches(img), seen)
    img[:, :, :] = torch.where(torch.isin(torch.arange(64), hidden).reshape(8, 8).repeat_interleave(4, 0).repeat_interleave(4, 1), torch.zeros(()), img)
    b = enc(jepa.patches(img), seen)                # the hidden patches painted black: nothing may change
    assert a.shape == (2, len(seen), jepa.EMB) and torch.allclose(a, b, atol=1e-5)


def test_predictor_gives_one_guess_for_each_hidden_patch():
    enc, pred = jepa.Encoder(), jepa.Predictor()
    img, _, _ = jepa.pictures(3, G())
    seen, hidden = jepa.hide(G(2))
    assert pred(enc(jepa.patches(img), seen), seen, hidden).shape == (3, len(hidden), jepa.EMB)


def test_round_cloud_rule_punishes_a_squeezed_cloud():
    cloud = torch.randn(1024, jepa.EMB, generator=G())
    assert jepa.roundness(cloud, G(1)) < 2                          # a round cloud scores about 1
    assert jepa.roundness(cloud * 0.01, G(1)) > 100                 # the same cloud squeezed to a point


def test_two_photos_of_a_scene_share_the_shape_but_not_the_speckle():
    scene = jepa.scenes(4, G())
    a, b = jepa.draw(*scene, G(1)), jepa.draw(*scene, G(2))
    assert ((a == 1) == (b == 1)).all() and not torch.equal(a, b)


def test_moves_stay_on_the_board():
    at = torch.tensor([[8, 8], [24, 24], [16, 16]])
    assert jepa.push(at, torch.tensor([3, 4, 1])).tolist() == [[8, 8], [24, 24], [16, 12]]


def test_planner_returns_one_of_the_five_moves():
    enc, mover = jepa.Encoder().eval(), jepa.Mover().eval()
    img = jepa.frame(torch.tensor([0, 0]), torch.tensor([[8, 8], [16, 16]]), G())
    move, far = jepa.plan(enc, mover, img[0], img[1], ahead=2)
    assert 0 <= move < 5 and len(far) == 25


if __name__ == '__main__':
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith('test_')]
    for n, f in tests:
        f()
        print('ok  ', n)
    print(f'{len(tests)} checks passed')
