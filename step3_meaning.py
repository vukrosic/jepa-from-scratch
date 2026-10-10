"""Step 3. Guess the hidden LISTS, not the hidden points. One encoder makes the question and the answer.
It finds a way to cheat: give every picture the same list."""
import jepa

enc, hist = jepa.trained('same')
print(f'{"round":>6}{"miss":>12}{"spread":>10}{"names the shape":>18}')
for step, miss, spread, right, _ in hist:
    print(f'{step:6}{miss:12.5f}{spread:10.4f}{right:16.1f} %')
print('\nThe miss is almost zero and the model has learned nothing. This is called collapse.')
