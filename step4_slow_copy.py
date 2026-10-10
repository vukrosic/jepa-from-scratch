"""Step 4. The repair from I-JEPA (2023): the answers come from a slow copy of the encoder.
The copy is never trained. After every round it moves 1 % of the way towards the encoder."""
import jepa

enc, hist = jepa.trained('slow')
print(f'{"round":>6}{"miss":>12}{"spread":>10}{"names the shape":>18}')
for step, miss, spread, right, _ in hist:
    print(f'{step:6}{miss:12.5f}{spread:10.4f}{right:16.1f} %')
