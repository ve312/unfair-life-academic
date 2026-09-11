import { InvalidStoryContentError } from '../errors/domain.error.js';
import { MAX_SHORT_DESCRIPTION_LENGTH, StorySummary } from './story-summary.entity.js';

describe('StorySummary', () => {
  it('builds a frozen summary and trims text', () => {
    const summary = new StorySummary({
      id: ' doc-1 ',
      title: ' Born in Guatemala ',
      shortDescription: ' A coffee-farming village. ',
      coverImageUrl: 'http://localhost:1337/uploads/village.jpg',
      year: 2026,
    });

    expect(summary).toMatchObject({
      id: 'doc-1',
      title: 'Born in Guatemala',
      shortDescription: 'A coffee-farming village.',
      coverImageUrl: 'http://localhost:1337/uploads/village.jpg',
      year: 2026,
    });
    expect(Object.isFrozen(summary)).toBe(true);
  });

  it('treats missing or blank optional fields as null', () => {
    const summary = new StorySummary({ id: 'doc-1', title: 'T', shortDescription: '   ', coverImageUrl: undefined });

    expect(summary.shortDescription).toBeNull();
    expect(summary.coverImageUrl).toBeNull();
    expect(summary.year).toBeNull();
  });

  it.each([
    ['a blank id', { id: ' ' }],
    ['a blank title', { title: '' }],
    ['a short description that is not text', { shortDescription: 42 as unknown as string }],
    ['a short description over the CMS limit', { shortDescription: 'x'.repeat(MAX_SHORT_DESCRIPTION_LENGTH + 1) }],
    ['an unsupported year', { year: 1850 }],
    ['a non-integer year', { year: 2026.5 }],
  ])('rejects %s', (_label, override) => {
    expect(() => new StorySummary({ id: 'doc-1', title: 'T', ...override })).toThrow(InvalidStoryContentError);
  });
});
