"""Exercise. The planner in step 6 reads the lists of the slow-copy encoder.
Swap in another encoder and play the same 200 games.

Write down your guess first:
  1. The encoder that was trained to paint points (step 2). Does the planner still reach the goal?
  2. The encoder that fell flat (step 3). All its lists are almost the same. What can the planner do?
  3. Which matters more for the planner: naming the shape, or knowing where the shape is?

Then run this file and compare."""
import jepa

print('200 games, 10 moves each, planner looks 1 move ahead.\n')
print(f'{"encoder":>10}  {"names the shape":>16}  {"centre off":>11}  {"games won":>10}')
for kind in ('pixels', 'same', 'slow', 'spread'):
    enc, _ = jepa.trained(kind)
    mover, _ = jepa.mover_for(kind, enc)
    right, off = jepa.probe(enc)
    print(f'{kind:>10}  {right:14.1f} %  {off:11.2f}  {jepa.play(enc, mover):8.1f} %')
