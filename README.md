# JEPA From Scratch

A small JEPA in 376 lines of PyTorch (`jepa.py`), trained on a laptop CPU.
A GPT guesses the next word. A JEPA guesses the *meaning* of a hidden part: a short list of numbers, not the hidden points themselves.
This repository builds that idea in six steps on tiny pictures (32 × 32 points, 8 shapes on speckle), breaks it, repairs it twice and then uses it to plan moves in a game.

Companion code for the video: **JEPA From Scratch: The Model That Predicts Meaning, Not Words** (link coming soon).

## Run it

```bash
pip install torch
sh run_all.sh        # every step and the tests
```

Python 3 and PyTorch, CPU only, no GPU needed. The first run trains every model (6,000 rounds each) and saves
them in `files/`; later runs load them. Run one step by itself with `python step4_slow_copy.py`.
The full output of one run is in [`demo-output.txt`](demo-output.txt).

## The course, file by file

| Step | File | What it does | Measured |
| --- | --- | --- | --- |
| 1. The world | `step1_pictures.py` | Pictures of one shape on speckle. Nobody tells the model what the shape is. The names are used only for the test: freeze the encoder, fit one layer on its lists, count the shapes named right. | chance 12.5 %, raw points 25.3 %, untrained encoder 30.5 % |
| 2. Guess points | `step2_pixels.py` | The GPT way for pictures: hide patches, paint the hidden points back. The speckle cannot be guessed, so the miss stops at a floor. | miss 0.061 (floor 0.041), names the shape 72.2 % |
| 3. Guess lists | `step3_meaning.py` | Guess the hidden *lists*. One encoder makes the question and the answer. It cheats: every picture gets the same list. | miss 0.00001, spread 0.001, names the shape 22.2 % |
| 4. Slow copy | `step4_slow_copy.py` | The repair from I-JEPA (2023): the answers come from a copy of the encoder that is never trained and moves 1 % towards the encoder after every round. | spread 0.84, names the shape 54.9 % |
| 5. Round cloud | `step5_spread.py` | The repair from LeJEPA (2025): no copy, no guesser. Rule one: four photos of one scene get one list. Rule two: the lists of many scenes form a round cloud. | spread 0.88, names the shape 62.7 % |
| 6. Decide | `step6_decide.py` | As in V-JEPA 2 (2025): a second network (the mover, 112,832 numbers) learns what each move does to the lists. The planner tries moves in its head and takes the one nearest to the lists of a photo of the goal. | 200 games of 10 moves: random 2.5 %, planner 90.0 % |

Everything the steps use is in `jepa.py`: the pictures, the encoder (155,264 numbers), the guesser (66,016 numbers),
the four ways to train, the test and the planner. `tests.py` holds 9 checks.

"Names the shape" is the test score on pictures the model never saw. "Miss" is the training loss.
"Spread" is how different the lists of different pictures are (0 means every picture gets the same list).

## Two results that surprised us

- **Points beat lists at naming shapes here** (72.2 % against 54.9 % and 62.7 %). In the I-JEPA paper it is the other way
  round on real photos (66.9 % against 40.7 %, the authors' own comparison). We did not test why. One likely reason:
  our speckle is simple and real photos hide far more detail.
- **Looking further ahead made the planner worse**: one move ahead 90.0 %, two moves 54.0 %, three moves 33.5 %.
  Every imagined step adds error and the errors pile up. The V-JEPA 2 authors name the same limit.

Lists still won at deciding: the same planner comparing photos point by point, with a perfect mover, won 82.0 %.

## How ours differs from the papers

| | here | the papers |
| --- | --- | --- |
| pictures | 32 × 32 drawn shapes on speckle | real photos (ImageNet), over 1 million hours of video |
| encoder | 155,264 numbers | up to about 1 billion |
| slow copy | moves 1 % a round | moves 0.4 % or less a round (I-JEPA) |
| round-cloud step | four photos of one drawn scene; a simple test of the cloud's shape | crops of real photos; SIGReg (LeJEPA) |
| planning | 5 moves on a 5 × 5 board, every move tried | a robot arm, sampled moves, about 16 s per move (V-JEPA 2) |

Papers: I-JEPA [arXiv 2301.08243](https://arxiv.org/abs/2301.08243), V-JEPA 2 [arXiv 2506.09985](https://arxiv.org/abs/2506.09985),
LeJEPA [arXiv 2511.08544](https://arxiv.org/abs/2511.08544). Their code: `facebookresearch/ijepa`, `facebookresearch/vjepa2`, `galilai-group/lejepa`.

## Exercise

The planner in step 6 reads the lists of the slow-copy encoder. `exercise.py` swaps in the other encoders and plays the same 200 games.
Write down your guess first, then run it:

```bash
python exercise.py
```

<details>
<summary>Answers (one run, seed 0)</summary>

| encoder | games won |
| --- | --- |
| trained to paint points (step 2) | 98.0 % |
| collapsed (step 3) | 0.0 % |
| slow copy (step 4) | 90.0 % |
| round cloud (step 5) | 75.5 % |

The collapsed encoder gives every picture almost the same list, so every move looks equally near to the goal.
The encoder that painted points gives the best planner on this board: the place of the shape is exactly what it had to keep.
That is a different question from the one in the video, which compares planning in lists with planning in raw photos point by point (82.0 %).
</details>

## Licence

The code is under the MIT licence, see `LICENSE`. The pictures are drawn by the program; no data set is downloaded.
