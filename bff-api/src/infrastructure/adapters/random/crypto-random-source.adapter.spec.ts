import { CryptoRandomSource } from './crypto-random-source.adapter.js';

describe('CryptoRandomSource', () => {
  it('returns numbers in [0, 1) spread across the whole interval', () => {
    const source = new CryptoRandomSource();
    const values = Array.from({ length: 20_000 }, () => source.next());

    expect(values.every((v) => v >= 0 && v < 1)).toBe(true);
    const mean = values.reduce((a, b) => a + b, 0) / values.length;
    expect(mean).toBeGreaterThan(0.48);
    expect(mean).toBeLessThan(0.52);
    const deciles: number[] = Array.from({ length: 10 }, () => 0);
    values.forEach((v) => deciles[Math.floor(v * 10)]++);
    deciles.forEach((count) => expect(count).toBeGreaterThan(1700));
  });
});
