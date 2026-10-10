"""Step 2. The GPT way, for pictures: hide a part, make the model paint the hidden points back."""
import jepa

enc, hist = jepa.trained('pixels')
print(f'{"round":>6}{"miss":>10}{"names the shape":>18}')
for step, miss, _, right, _ in hist:
    print(f'{step:6}{miss:10.4f}{right:16.1f} %')
print(f'\nThe background is random. Nothing can guess it. A model that always paints the middle grey')
print(f'would miss by {jepa.NOISE ** 2 / 12:.4f} on a background point, so most of the miss can never go away.')
