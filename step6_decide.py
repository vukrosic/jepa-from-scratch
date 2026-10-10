"""Step 6. From seeing to deciding, as in V-JEPA 2 (2025). A shape sits on a 5 x 5 board. Five moves.
A second network learns what each move does to the LISTS. To decide, the planner tries moves in its head
and takes the one whose guessed lists are nearest to the lists of a picture of the goal."""
import jepa

enc, _ = jepa.trained('slow')
mover, hist = jepa.mover_for('slow', enc)
print(f'The mover: {jepa.count(mover):,} numbers. Its miss: {hist[0][1]:.4f} at the start, {hist[-1][1]:.4f} after {hist[-1][0]:,} rounds.\n')
print(f'200 games, 10 moves each. A game is won when the shape ends on the goal.')
print(f'  random moves:                 {jepa.play(enc, mover, pick="random"):5.1f} %')
for ahead in (1, 2, 3):
    print(f'  planner, looks {ahead} move{"s" if ahead > 1 else " "} ahead:  {jepa.play(enc, mover, ahead=ahead):5.1f} %')
print(f'\nFor comparison, a planner that compares the pictures point by point')
print(f'and gets a perfect picture of what each move does:  {jepa.play(None, None, pick="points"):5.1f} %')
