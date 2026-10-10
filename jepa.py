"""A small JEPA in PyTorch: the pictures, the networks, four ways to train them, the test and the planner.
Every step file imports this."""
import os, json, copy, hashlib, itertools
import torch, torch.nn as nn, torch.nn.functional as F

torch.set_num_threads(4)
HERE = os.path.dirname(os.path.abspath(__file__))
FILES = os.path.join(HERE, 'files')

# ---------------------------------------------------------------- the pictures
SIZE, PATCH = 32, 4                 # a picture is 32 x 32 points, cut into patches of 4 x 4
GRID = SIZE // PATCH                # 8 x 8 = 64 patches
SHAPES = ['square', 'disc', 'triangle', 'cross', 'ring', 'bar', 'diamond', 'corner']
NOISE = 0.7                         # every background point is a random grey between 0 and 0.7
EMB = 64                            # the encoder describes each patch with a list of 64 numbers


def draw(kind, cx, cy, r, gen):
    """One white shape on a noisy background for each entry. kind, cx, cy, r are tensors of the same length."""
    n = len(kind)
    ys, xs = torch.meshgrid(torch.arange(SIZE).float(), torch.arange(SIZE).float(), indexing='ij')
    u = (xs[None] - cx[:, None, None]) / r[:, None, None]
    v = (ys[None] - cy[:, None, None]) / r[:, None, None]
    a, b, d = u.abs(), v.abs(), (u * u + v * v).sqrt()
    box = torch.maximum(a, b)
    every = torch.stack([
        box <= 0.8,                                     # square
        d <= 1.0,                                       # disc
        (b <= 0.9) & (a <= (v + 0.9) / 1.8),            # triangle
        ((a <= 0.3) | (b <= 0.3)) & (box <= 1.0),       # cross
        (d <= 1.0) & (d >= 0.6),                        # ring
        (b <= 0.35) & (a <= 1.0),                       # bar
        a + b <= 1.0,                                   # diamond
        (box <= 0.9) & ((u <= -0.3) | (v >= 0.3)),      # corner
    ], 1)
    shape = every[torch.arange(n), kind]
    noise = torch.rand(n, SIZE, SIZE, generator=gen) * NOISE
    return torch.where(shape, torch.ones(()), noise)


def scenes(n, gen):
    """n random scenes: which shape, where its centre is, how big it is."""
    kind = torch.randint(len(SHAPES), (n,), generator=gen)
    r = 5 + 2 * torch.rand(n, generator=gen)
    cx, cy = 8 + 16 * torch.rand(n, generator=gen), 8 + 16 * torch.rand(n, generator=gen)
    return kind, cx, cy, r


def pictures(n, gen):
    """n random pictures. Also returns which shape is in each and where its centre is (used only for testing)."""
    kind, cx, cy, r = scenes(n, gen)
    return draw(kind, cx, cy, r, gen), kind, torch.stack([cx, cy], 1)


def patches(img):
    """n x 32 x 32 pictures -> n x 64 patches x 16 points."""
    n = len(img)
    return img.reshape(n, GRID, PATCH, GRID, PATCH).permute(0, 1, 3, 2, 4).reshape(n, GRID * GRID, PATCH * PATCH)


def hide(gen):
    """Hide two blocks of 3 x 3 patches. Returns the patches the model may see and the hidden ones."""
    hidden = set()
    for _ in range(2):
        r, c = torch.randint(GRID - 2, (2,), generator=gen).tolist()
        hidden |= {(r + i) * GRID + c + j for i in range(3) for j in range(3)}
    seen = [i for i in range(GRID * GRID) if i not in hidden]
    return torch.tensor(seen), torch.tensor(sorted(hidden))


# ---------------------------------------------------------------- the networks
def layer(width):
    """One attention layer, the same kind a GPT is made of."""
    return nn.TransformerEncoderLayer(width, 4, 4 * width, dropout=0.0, activation='gelu', batch_first=True, norm_first=True)


class Encoder(nn.Module):
    """Patches in, one list of 64 numbers for each patch out."""

    def __init__(self, layers=3):
        super().__init__()
        self.patch = nn.Linear(PATCH * PATCH, EMB)
        self.place = nn.Parameter(torch.randn(GRID * GRID, EMB) * 0.02)     # where each patch sits
        self.layers = nn.ModuleList([layer(EMB) for _ in range(layers)])
        self.norm = nn.LayerNorm(EMB)

    def forward(self, p, seen=None):
        x = self.patch(p) + self.place
        if seen is not None:
            x = x[:, seen]                  # the hidden patches are removed before the network runs
        for l in self.layers:
            x = l(x)
        return self.norm(x)


class Predictor(nn.Module):
    """The lists of the seen patches in, a guess for each hidden patch out."""

    def __init__(self, out=EMB, width=48, layers=2):
        super().__init__()
        self.inp = nn.Linear(EMB, width)
        self.blank = nn.Parameter(torch.zeros(width))                       # stands in for a hidden patch
        self.place = nn.Parameter(torch.randn(GRID * GRID, width) * 0.02)
        self.layers = nn.ModuleList([layer(width) for _ in range(layers)])
        self.norm = nn.LayerNorm(width)
        self.out = nn.Linear(width, out)

    def forward(self, lists, seen, hidden):
        x = self.inp(lists) + self.place[seen]
        ask = (self.blank + self.place[hidden]).expand(len(x), -1, -1)      # "what is at this place?"
        x = torch.cat([x, ask], 1)
        for l in self.layers:
            x = l(x)
        return self.out(self.norm(x[:, len(seen):]))


def count(*models):
    return sum(p.numel() for m in models for p in m.parameters())


# ---------------------------------------------------------------- the round-cloud rule (step 5)
def roundness(z, gen, lines=128):
    """How far a cloud of lists is from a round bell-shaped cloud. 0 means it is one.
    Look at the cloud along many random directions. Along each, compare the points with a bell curve."""
    way = torch.randn(z.shape[1], lines, generator=gen)
    way = way / way.norm(dim=0)
    x = z @ way                                         # every list, seen along every direction
    t = torch.linspace(-5, 5, 17)
    xt = x[:, :, None] * t
    bell = torch.exp(-t * t / 2)
    miss = (xt.cos().mean(0) - bell) ** 2 + xt.sin().mean(0) ** 2
    return torch.trapezoid(miss * bell, t).mean() * len(z)


# ---------------------------------------------------------------- training
KINDS = ('pixels', 'same', 'slow', 'spread', 'labels')


def train(kind, steps=6000, seed=0, size=128, lr=1e-3, slow=0.99, weight=0.05, log=250, say=False, keep=None):
    """pixels: guess the hidden points.          same:   guess the hidden lists, one encoder makes both sides.
    slow:   the answers come from a slow copy.   spread: four photos of a scene must get one list, and the cloud of lists must be round.
    labels: a teacher names the shape (the usual way, for comparison)."""
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    enc = Encoder()
    pred = Predictor(out=PATCH * PATCH if kind == 'pixels' else EMB)
    copy_ = copy.deepcopy(enc) if kind == 'slow' else None
    head = nn.Linear(EMB, len(SHAPES))
    short = nn.Linear(EMB, 16)                                          # spread only: 16 numbers for a whole picture
    parts = {'labels': [enc, head], 'spread': [enc, short]}.get(kind, [enc, pred])
    opt = torch.optim.Adam([p for m in parts for p in m.parameters()], lr=lr)
    hist = []
    for step in range(steps + 1):
        if kind == 'spread':
            # four photos of each scene: same shape, same place, new speckle, other patches hidden
            scene = scenes(size, gen)
            z = torch.stack([short(enc(patches(draw(*scene, gen)), hide(gen)[0]).mean(1)) for _ in range(4)])
            agree = (z - z.mean(0)).pow(2).mean()                       # the four lists should be one list
            loss = (1 - weight) * agree + weight * sum(roundness(v, gen) for v in z) / 4
        else:
            img, label, _ = pictures(size, gen)
            p = patches(img)
            seen, hidden = hide(gen)
        if kind == 'labels':
            loss = F.cross_entropy(head(enc(p).mean(1)), label)
        elif kind != 'spread':
            guess = pred(enc(p, seen), seen, hidden)
            if kind == 'pixels':
                loss = F.mse_loss(guess, p[:, hidden])
            elif kind == 'slow':
                with torch.no_grad():
                    answer = copy_(p)[:, hidden]
                loss = F.mse_loss(guess, answer)
            else:
                loss = F.mse_loss(guess, enc(p)[:, hidden])
        if step % log == 0:
            judge = copy_ if kind == 'slow' else enc
            right, off = probe(judge)
            hist.append([step, round(loss.item(), 5), round(spread(judge), 5), right, off])
            if say:
                print(f'{step:6}  miss {loss.item():9.5f}   spread {spread(judge):7.4f}   names the shape {right:5.1f} %', flush=True)
        if step == steps:
            break
        opt.zero_grad()
        loss.backward()
        opt.step()
        if kind == 'slow':
            with torch.no_grad():
                for a, b in zip(copy_.parameters(), enc.parameters()):
                    a.mul_(slow).add_(b, alpha=1 - slow)        # the copy moves 1 % of the way each round
    if keep is not None:
        keep['guesser'] = pred
    return (copy_ if kind == 'slow' else enc), hist


def trained(kind, **kw):
    """Train once, then keep the result in files/."""
    key = hashlib.sha1(json.dumps([kind, sorted(kw.items())]).encode()).hexdigest()[:10]
    base = os.path.join(FILES, f'{kind}-{key}')
    enc = Encoder()
    if os.path.exists(base + '.pt'):
        enc.load_state_dict(torch.load(base + '.pt'))
        return enc.eval(), json.load(open(base + '.json'))
    keep = {}
    enc, hist = train(kind, say=True, keep=keep, **kw)
    os.makedirs(FILES, exist_ok=True)
    torch.save(enc.state_dict(), base + '.pt')
    if kind == 'pixels':                                # the painter is kept, so we can look at what it paints
        torch.save(keep['guesser'].state_dict(), os.path.join(FILES, 'painter.pt'))
    json.dump(hist, open(base + '.json', 'w'))
    return enc.eval(), hist


def painter():
    """The guesser of the points model (step 2)."""
    trained('pixels')
    g = Predictor(out=PATCH * PATCH)
    g.load_state_dict(torch.load(os.path.join(FILES, 'painter.pt')))
    return g.eval()


# ---------------------------------------------------------------- the test
_TEST = {}


def test_set():
    """6,000 pictures the models never train on: 4,000 to fit the reader, 2,000 to score it."""
    if not _TEST:
        img, kind, where = pictures(6000, torch.Generator().manual_seed(999))
        _TEST.update(img=img, kind=kind, where=where)
    return _TEST['img'], _TEST['kind'], _TEST['where']


@torch.no_grad()
def describe(enc, img):
    """One list of 64 numbers for a whole picture: the average of its 64 patch lists."""
    return torch.cat([enc(patches(img[i:i + 500])).mean(1) for i in range(0, len(img), 500)])


@torch.no_grad()
def spread(enc):
    """How different the lists of different pictures are. 0 means every picture gets the same list."""
    return describe(enc, test_set()[0][:500]).std(0).mean().item()


@torch.no_grad()
def probe(enc, floor=1e-4):
    """Freeze the encoder. Fit ONE straight-line layer on its lists with 4,000 named pictures, score it on 2,000 others.
    Returns: shape named right (%), and how far off it puts the centre (points). enc=None reads the raw points."""
    img, kind, where = test_set()
    x = img.reshape(len(img), -1) if enc is None else describe(enc, img)
    x = (x - x[:4000].mean(0)) / max(x[:4000].std().item(), floor)      # below the floor, differences count as none
    x = torch.cat([x, torch.ones(len(x), 1)], 1).double()
    y = torch.cat([F.one_hot(kind, len(SHAPES)).double(), where.double()], 1)
    a, b = x[:4000], y[:4000]
    w = torch.linalg.solve(a.T @ a + torch.eye(x.shape[1]).double(), a.T @ b)
    out = x[4000:] @ w
    right = (out[:, :len(SHAPES)].argmax(1) == kind[4000:]).float().mean().item() * 100
    off = (out[:, len(SHAPES):] - where[4000:]).norm(dim=1).mean().item()
    return round(right, 1), round(off, 2)


# ---------------------------------------------------------------- deciding (step 6)
MOVES = {'stay': (0, 0), 'up': (0, -4), 'down': (0, 4), 'left': (-4, 0), 'right': (4, 0)}
NAMES = list(MOVES)
STEP = torch.tensor(list(MOVES.values()))
CELLS = (8, 12, 16, 20, 24)         # the shape sits on a 5 x 5 board


def frame(kind, at, gen):
    """A picture of the board. The noise is new in every frame."""
    return draw(kind, at[:, 0].float(), at[:, 1].float(), torch.full((len(kind),), 6.0), gen)


def push(at, move):
    return (at + STEP[move]).clamp(CELLS[0], CELLS[-1])


class Mover(nn.Module):
    """The lists of one frame and a move in, a guess for the lists of the next frame out."""

    def __init__(self, width=64, layers=2):
        super().__init__()
        self.inp = nn.Linear(EMB, width)
        self.place = nn.Parameter(torch.randn(GRID * GRID, width) * 0.02)
        self.move = nn.Embedding(len(MOVES), width)
        self.layers = nn.ModuleList([layer(width) for _ in range(layers)])
        self.norm = nn.LayerNorm(width)
        self.out = nn.Linear(width, EMB)

    def forward(self, z, move):
        x = self.inp(z) + self.place + self.move(move)[:, None]
        for l in self.layers:
            x = l(x)
        return self.out(self.norm(x))


def train_mover(enc, steps=2000, seed=0, size=128, lr=1e-3, say=False):
    """The encoder stays frozen. The mover watches random moves and learns what each one does to the lists."""
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    mover = Mover()
    opt = torch.optim.Adam(mover.parameters(), lr=lr)
    hist = []
    for step in range(steps + 1):
        kind = torch.randint(len(SHAPES), (size,), generator=gen)
        at = torch.tensor(CELLS)[torch.randint(5, (size, 2), generator=gen)]
        move = torch.randint(len(MOVES), (size,), generator=gen)
        with torch.no_grad():
            now = enc(patches(frame(kind, at, gen)))
            then = enc(patches(frame(kind, push(at, move), gen)))
        loss = (mover(now, move) - then).abs().mean()
        if step % 250 == 0:
            hist.append([step, round(loss.item(), 4)])
            if say:
                print(f'{step:6}  miss {loss.item():.4f}', flush=True)
        if step == steps:
            break
        opt.zero_grad()
        loss.backward()
        opt.step()
    return mover.eval(), hist


def mover_for(name, enc, **kw):
    key = hashlib.sha1(json.dumps([name, sorted(kw.items())]).encode()).hexdigest()[:10]
    base = os.path.join(FILES, f'mover-{key}')
    if os.path.exists(base + '.pt'):
        m = Mover()
        m.load_state_dict(torch.load(base + '.pt'))
        return m.eval(), json.load(open(base + '.json'))
    m, hist = train_mover(enc, say=True, **kw)
    os.makedirs(FILES, exist_ok=True)
    torch.save(m.state_dict(), base + '.pt')
    json.dump(hist, open(base + '.json', 'w'))
    return m, hist


@torch.no_grad()
def plan(enc, mover, now_img, goal_img, ahead=1):
    """Try every chain of `ahead` moves in the head. Return the first move of the chain that ends nearest the goal.
    Nearest is measured between lists, never between pictures."""
    z, goal = enc(patches(now_img[None])), enc(patches(goal_img[None]))
    chains = torch.tensor(list(itertools.product(range(len(MOVES)), repeat=ahead)))
    z = z.expand(len(chains), -1, -1)
    for k in range(ahead):
        z = mover(z, chains[:, k])
    far = (z - goal).abs().mean((1, 2))
    return chains[far.argmin(), 0].item(), far


@torch.no_grad()
def play(enc, mover, games=200, moves=10, ahead=1, seed=0, pick=None):
    """Put a shape on the board, show a picture of where it should be, let the planner move it.
    Returns the share of games that end on the goal (%). pick='random' plays random moves,
    pick='points' compares whole pictures point by point (enc and mover are not used then)."""
    gen = torch.Generator().manual_seed(seed)
    won = 0
    for g in range(games):
        kind = torch.randint(len(SHAPES), (1,), generator=gen)
        at = torch.tensor(CELLS)[torch.randint(5, (1, 2), generator=gen)]
        goal = at
        while (goal == at).all():
            goal = torch.tensor(CELLS)[torch.randint(5, (1, 2), generator=gen)]
        goal_img = frame(kind, goal, gen)[0]
        for _ in range(moves):
            if pick == 'random':
                m = torch.randint(len(MOVES), (1,), generator=gen).item()
            elif pick == 'points':      # compare pictures point by point, with a perfect picture of what each move does
                far = [(frame(kind, push(at, torch.tensor([k])), gen)[0] - goal_img).abs().mean().item() for k in range(len(MOVES))]
                m = min(range(len(MOVES)), key=far.__getitem__)
            else:
                m, _ = plan(enc, mover, frame(kind, at, gen)[0], goal_img, ahead)
            at = push(at, torch.tensor([m]))
        won += int((at == goal).all())
    return round(100 * won / games, 1)
