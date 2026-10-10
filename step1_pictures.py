"""Step 1. The pictures and the test. Nobody tells the model what is in a picture.
The names of the shapes are used for one thing only: to test what a model has learned."""
import torch, jepa

img, kind, where = jepa.pictures(3, torch.Generator().manual_seed(4))
for i in range(3):
    print(f'A {jepa.SHAPES[kind[i]]} at {where[i, 0]:.0f}, {where[i, 1]:.0f}:')
    for row in img[i][::2]:
        print('  ' + ''.join('#' if v == 1 else '.:'[int(v > jepa.NOISE / 2)] for v in row))
print(f'\nA picture is {jepa.SIZE} x {jepa.SIZE} = {jepa.SIZE ** 2} points, cut into {jepa.GRID ** 2} patches of {jepa.PATCH} x {jepa.PATCH}.')
print(f'{len(jepa.SHAPES)} shapes, so a blind guess names the shape right {100 / len(jepa.SHAPES):.1f} % of the time.\n')

print('The test: freeze the model, fit one straight-line layer on what it gives, name the shape in 2,000 new pictures.')
print(f'  reading the raw points:              {jepa.probe(None)[0]:5.1f} %')
torch.manual_seed(0)
enc = jepa.Encoder()
print(f'  an encoder with random numbers:      {jepa.probe(enc)[0]:5.1f} %   ({jepa.count(enc):,} numbers)')
