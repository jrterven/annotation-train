// Independent fixtures are encoded by official pycocotools, row-major expected
// pixels ensure the browser's COCO decoder cannot silently transpose masks.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { decodeCounts, maskContains } from '../frontend/src/masks.ts';

const fixtures = JSON.parse(readFileSync(new URL('./fixtures/rle-masks.json', import.meta.url), 'utf8'));
for (const fixture of fixtures) {
  test(`COCO browser decoder: ${fixture.name}`, () => {
    const { mask } = fixture;
    const [height, width] = mask.size;
    const expected = Buffer.from(fixture.pixels, 'base64');
    assert.equal(decodeCounts(mask.counts).reduce((a, b) => a + b, 0), width * height);
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        assert.equal(maskContains(mask, x, y), !!expected[y * width + x], `${x},${y}`);
      }
    }
    assert.equal(maskContains(mask, -1, 0), false);
    assert.equal(maskContains(mask, width, height - 1), false);
  });
}
test('Malformed compressed masks are rejected', () => {
  assert.throws(() => decodeCounts('!'));
  assert.throws(() => decodeCounts('o'));
});
