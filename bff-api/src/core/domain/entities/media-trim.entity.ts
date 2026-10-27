import { InvalidStoryContentError } from '../errors/domain.error.js';

/** Upper bound for a fragment boundary (24 hours), to catch typos such as milliseconds entered as seconds. */
export const MAX_MEDIA_TIME_SECONDS = 86_400;

export interface MediaTrimProps {
  startTime?: number | null;
  endTime?: number | null;
}

/**
 * The fragment of a media file a scene plays, in seconds from the file's start.
 * Either bound may be absent: no start plays from 0, no end plays to the
 * file's natural end. The client trims playback; the file itself is untouched.
 */
export class MediaTrim {
  static readonly NONE = new MediaTrim({});

  readonly startTime: number | null;
  readonly endTime: number | null;

  constructor(props: MediaTrimProps, context = 'Media trim') {
    this.startTime = MediaTrim.bound('startTime', props.startTime, context);
    this.endTime = MediaTrim.bound('endTime', props.endTime, context);
    if (this.endTime !== null && this.endTime <= (this.startTime ?? 0)) {
      throw new InvalidStoryContentError(
        `${context}: 'endTime' (${this.endTime}s) must be after 'startTime' (${this.startTime ?? 0}s)`,
      );
    }
    Object.freeze(this);
  }

  /** True when the whole file plays. */
  get isFullLength(): boolean {
    return this.startTime === null && this.endTime === null;
  }

  private static bound(name: string, value: unknown, context: string): number | null {
    if (value === null || value === undefined) {
      return null;
    }
    if (typeof value !== 'number' || !Number.isFinite(value) || value < 0 || value > MAX_MEDIA_TIME_SECONDS) {
      throw new InvalidStoryContentError(
        `${context}: '${name}' must be a number of seconds between 0 and ${MAX_MEDIA_TIME_SECONDS}, got ${typeof value === 'number' ? value : JSON.stringify(value)}`,
      );
    }
    return value;
  }
}
