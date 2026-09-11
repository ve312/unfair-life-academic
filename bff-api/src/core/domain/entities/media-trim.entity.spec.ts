import { InvalidStoryContentError } from '../errors/domain.error.js';
import { MAX_MEDIA_TIME_SECONDS, MediaTrim } from './media-trim.entity.js';

describe('MediaTrim', () => {
  it.each([
    [{ startTime: 5, endTime: 12.5 }, 5, 12.5],
    [{ startTime: 5 }, 5, null],
    [{ endTime: 8 }, null, 8],
    [{ startTime: 0, endTime: 0.5 }, 0, 0.5],
  ])('accepts %j', (props, start, end) => {
    const trim = new MediaTrim(props);

    expect(trim.startTime).toBe(start);
    expect(trim.endTime).toBe(end);
    expect(trim.isFullLength).toBe(false);
    expect(Object.isFrozen(trim)).toBe(true);
  });

  it('NONE plays the whole file', () => {
    expect(MediaTrim.NONE).toMatchObject({ startTime: null, endTime: null, isFullLength: true });
  });

  it.each([
    ['an end before the start', { startTime: 10, endTime: 4 }],
    ['an end equal to the start', { startTime: 3, endTime: 3 }],
    ['an end of 0 without a start', { endTime: 0 }],
    ['a negative start', { startTime: -0.1 }],
    ['an infinite end', { endTime: Number.POSITIVE_INFINITY }],
    ['NaN', { startTime: Number.NaN }],
    ['more than 24 hours (milliseconds typed as seconds)', { startTime: MAX_MEDIA_TIME_SECONDS + 1 }],
  ])('rejects %s', (_label, props) => {
    expect(() => new MediaTrim(props as never, "Scene 'x': media trim")).toThrow(InvalidStoryContentError);
  });
});
