"""Step 5. The repair from LeJEPA (2025): no slow copy and no guesser. One encoder, two rules.
Rule 1: four photos of the same scene (same shape, same place, new speckle, other patches hidden) must get one list.
Rule 2: the lists of many scenes must form a round cloud. A cloud squeezed into one point breaks rule 2."""
import jepa

enc, hist = jepa.trained('spread')
print(f'{"round":>6}{"miss":>12}{"spread":>10}{"names the shape":>18}')
for step, miss, spread, right, _ in hist:
    print(f'{step:6}{miss:12.5f}{spread:10.4f}{right:16.1f} %')

print(f'\n{"after 6,000 rounds":36}{"names the shape":>16}')
rows = [('raw points', jepa.probe(None)[0])]
for kind, label in (('pixels', 'guess the points'), ('same', 'guess the lists, one encoder'), ('slow', 'slow copy'), ('spread', 'one list for a scene + round cloud'), ('labels', 'a teacher names every shape')):
    rows.append((label, jepa.trained(kind)[1][-1][3]))
for label, right in rows:
    print(f'{label:36}{right:14.1f} %')
