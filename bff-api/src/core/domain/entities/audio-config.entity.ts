import { InvalidStoryContentError } from '../errors/domain.error.js';

export interface AudioConfigProps {
  url: string;
  loop: boolean;
  defaultVolume: number;
}

/**
 * How a scene's audio track should be played by the client (Howler.js):
 * where to load it from, whether it loops, and its volume between 0 and 1.
 */
export class AudioConfig {
  readonly url: string;
  readonly loop: boolean;
  readonly defaultVolume: number;

  constructor(props: AudioConfigProps) {
    if (typeof props.url !== 'string' || props.url.trim().length === 0) {
      throw new InvalidStoryContentError(`Audio config: 'url' must be a non-empty string`);
    }
    if (typeof props.loop !== 'boolean') {
      throw new InvalidStoryContentError(`Audio config '${props.url}': 'loop' must be a boolean`);
    }
    if (
      typeof props.defaultVolume !== 'number' ||
      !Number.isFinite(props.defaultVolume) ||
      props.defaultVolume < 0 ||
      props.defaultVolume > 1
    ) {
      throw new InvalidStoryContentError(`Audio config '${props.url}': 'defaultVolume' must be a number between 0 and 1`);
    }
    this.url = props.url.trim();
    this.loop = props.loop;
    this.defaultVolume = props.defaultVolume;
    Object.freeze(this);
  }
}
