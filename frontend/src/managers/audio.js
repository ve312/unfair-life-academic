import { Howl, Howler } from 'howler';

const DEFAULT_FADE_MS = 800;
/** Extra time after a fade before forcing the stop, in case Howler never reports the fade's end. */
const FADE_SAFETY_MARGIN_MS = 150;
/** How often a trimmed track's position is checked against its end time. */
const SEGMENT_CHECK_MS = 50;

/**
 * @typedef {object} Segment Fragment of a file to play, in seconds.
 * @property {number} start
 * @property {number | null} end null plays to the file's natural end.
 */

/**
 * Plays one background track at a time with Howler.js.
 *
 * Starting a track stops the previous one first. With fadeMs > 0 the old
 * track fades out while the new one fades in (a short cross-fade); with
 * fadeMs = 0 it is a hard cut. At most one outgoing track is ever fading: if
 * tracks change again mid-fade, the older fading track is stopped at once,
 * so music never piles up.
 *
 * A track can be trimmed to a fragment (startTime / endTime, in seconds).
 * Playback starts at startTime; at endTime (or the file's end) a looping
 * track jumps back to startTime and a non-looping one fades out. Files are
 * never cut: the manager seeks and watches the position instead.
 *
 * Requesting the track that is already playing, with the same fragment, does
 * not restart it; only its loop flag and volume are updated, so consecutive
 * scenes that share a track play it seamlessly.
 */
export class AudioManager {
  /** @type {{ key: string, url: string, howl: Howl, segment: Segment | null, loop: boolean } | null} */
  #current = null;
  /** @type {Map<Howl, { key: string, url: string, segment: Segment | null, loop: boolean, timer: ReturnType<typeof setTimeout> }>} */
  #fadingOut = new Map();
  /** @type {Map<Howl, ReturnType<typeof setInterval>>} */
  #segmentMonitors = new Map();

  /**
   * @param {{ fadeMs?: number, onError?: (error: Error, url: string) => void }} [options]
   *   onError is called when a track fails to load or play; it defaults to console.error.
   */
  constructor({ fadeMs = DEFAULT_FADE_MS, onError } = {}) {
    if (!Number.isInteger(fadeMs) || fadeMs < 0) {
      throw new TypeError(`AudioManager: fadeMs must be a non-negative integer, got '${fadeMs}'`);
    }
    this.fadeMs = fadeMs;
    this.onError = onError ?? ((error, url) => console.error(`[AudioManager] ${url}:`, error));
  }

  /** URL of the track currently playing (or starting), or null. */
  get currentUrl() {
    return this.#current?.url ?? null;
  }

  /** Fragment of the current track ({ start, end } in seconds), or null when it plays whole. */
  get currentSegment() {
    return this.#current?.segment ? { ...this.#current.segment } : null;
  }

  /** Playback position of the current track in seconds, or null when nothing plays. */
  get currentPosition() {
    const position = this.#current?.howl.seek();
    return typeof position === 'number' ? position : null;
  }

  /**
   * Plays a track, stopping the previous one so tracks never overlap beyond the cross-fade.
   * @param {string} url Absolute or relative URL of the audio file.
   * @param {boolean} [loop=true] Loops the whole file, or the fragment when one is given.
   * @param {number} [volume=1] Target volume between 0 and 1 (values outside are clamped).
   * @param {{ startTime?: number | null, endTime?: number | null }} [trim] Fragment to play, in seconds.
   */
  playTrack(url, loop = true, volume = 1, { startTime = null, endTime = null } = {}) {
    if (typeof url !== 'string' || url.trim() === '') {
      throw new TypeError('playTrack: url must be a non-empty string');
    }
    if (typeof loop !== 'boolean') {
      throw new TypeError('playTrack: loop must be a boolean');
    }
    if (typeof volume !== 'number' || !Number.isFinite(volume)) {
      throw new TypeError('playTrack: volume must be a finite number');
    }
    const trackUrl = url.trim();
    const targetVolume = Math.min(1, Math.max(0, volume));
    const segment = AudioManager.#toSegment(startTime, endTime, trackUrl);
    const key = AudioManager.#keyFor(trackUrl, segment);

    // Same track and fragment already playing: keep it going, just apply the new settings.
    if (this.#current?.key === key) {
      this.#current.loop = loop;
      if (segment === null) {
        this.#current.howl.loop(loop);
      }
      this.#fadeTo(this.#current.howl, targetVolume);
      return;
    }

    // Same track is fading out (e.g. A -> B -> A quickly): bring it back instead of reloading it.
    const revived = this.#reviveFadingTrack(key);

    this.#stopAllFading();
    this.stopCurrentTrack();

    if (revived) {
      if (segment === null) {
        revived.loop(loop);
      }
      this.#current = { key, url: trackUrl, howl: revived, segment, loop };
      this.#fadeTo(revived, targetVolume);
      return;
    }

    const howl = this.#createHowl(trackUrl, segment === null && loop, this.fadeMs > 0 ? 0 : targetVolume);
    this.#current = { key, url: trackUrl, howl, segment, loop };
    const soundId = howl.play();
    if (segment !== null) {
      if (segment.start > 0) {
        // Howler queues the seek until the file can play, so playback begins at the fragment.
        howl.seek(segment.start, soundId);
      }
      this.#watchSegment(howl);
    }
    if (this.fadeMs > 0) {
      howl.fade(0, targetVolume, this.fadeMs);
    }
  }

  /**
   * Stops the current track, fading it out unless fadeMs is 0.
   * @param {{ fadeMs?: number }} [options]
   */
  stopCurrentTrack({ fadeMs = this.fadeMs } = {}) {
    const current = this.#current;
    if (current === null) {
      return;
    }
    this.#current = null;

    const { howl } = current;
    if (fadeMs <= 0 || !howl.playing()) {
      this.#dispose(howl);
      return;
    }

    const finish = () => {
      const entry = this.#fadingOut.get(howl);
      if (entry === undefined) {
        return; // Already disposed, or revived by playTrack.
      }
      clearTimeout(entry.timer);
      this.#fadingOut.delete(howl);
      this.#dispose(howl);
    };
    this.#fadingOut.set(howl, {
      key: current.key,
      url: current.url,
      segment: current.segment,
      loop: current.loop,
      timer: setTimeout(finish, fadeMs + FADE_SAFETY_MARGIN_MS),
    });
    howl.once('fade', finish);
    howl.fade(howl.volume(), 0, fadeMs);
  }

  /** Mutes or unmutes all game audio without stopping playback. */
  setMuted(muted) {
    Howler.mute(Boolean(muted));
  }

  /** Stops and releases every track this manager owns (e.g. when the game shuts down). */
  destroy() {
    this.#stopAllFading();
    this.stopCurrentTrack({ fadeMs: 0 });
  }

  #createHowl(url, loop, initialVolume) {
    const howl = new Howl({
      src: [url],
      // Streams long music files instead of downloading and decoding them fully first.
      html5: true,
      loop,
      volume: initialVolume,
      preload: true,
      onloaderror: (_soundId, error) => {
        this.#handleFailure(howl, url, new Error(`Audio failed to load: ${describeHowlerError(error)}`));
      },
      onplayerror: () => {
        // Browsers block audio until the player interacts with the page. This is expected on
        // the first track, so it is not reported: Howler fires 'unlock' on the first
        // click/tap/key press and the track starts then (if it is still the current one).
        howl.once('unlock', () => {
          if (this.#current?.howl === howl && !howl.playing()) {
            howl.play();
          }
        });
      },
    });
    return howl;
  }

  /**
   * Keeps a trimmed track inside its fragment: at the end time, or when the file
   * itself ends, it loops back to the start or fades out.
   */
  #watchSegment(howl) {
    const onFragmentEnd = () => {
      const owner = this.#current?.howl === howl ? this.#current : this.#fadingOut.get(howl);
      if (!owner || owner.segment === null) {
        return;
      }
      if (owner.loop) {
        howl.seek(owner.segment.start);
        if (!howl.playing()) {
          howl.play();
        }
      } else if (this.#current?.howl === howl) {
        this.stopCurrentTrack();
      } else {
        howl.stop();
      }
    };
    howl.on('end', onFragmentEnd);
    const monitor = setInterval(() => {
      const owner = this.#current?.howl === howl ? this.#current : this.#fadingOut.get(howl);
      const end = owner?.segment?.end;
      if (end == null || !howl.playing()) {
        return;
      }
      const position = howl.seek();
      if (typeof position === 'number' && position >= end) {
        onFragmentEnd();
      }
    }, SEGMENT_CHECK_MS);
    this.#segmentMonitors.set(howl, monitor);
  }

  #handleFailure(howl, url, error) {
    if (this.#current?.howl === howl) {
      this.#current = null;
    }
    const entry = this.#fadingOut.get(howl);
    if (entry !== undefined) {
      clearTimeout(entry.timer);
      this.#fadingOut.delete(howl);
    }
    this.#dispose(howl);
    this.onError(error, url);
  }

  #fadeTo(howl, targetVolume) {
    if (this.fadeMs > 0 && howl.playing()) {
      howl.fade(howl.volume(), targetVolume, this.fadeMs);
    } else {
      howl.volume(targetVolume);
    }
  }

  #reviveFadingTrack(key) {
    for (const [howl, entry] of this.#fadingOut) {
      if (entry.key === key) {
        clearTimeout(entry.timer);
        this.#fadingOut.delete(howl);
        return howl;
      }
    }
    return null;
  }

  #stopAllFading() {
    for (const [howl, entry] of this.#fadingOut) {
      clearTimeout(entry.timer);
      this.#dispose(howl);
    }
    this.#fadingOut.clear();
  }

  #dispose(howl) {
    const monitor = this.#segmentMonitors.get(howl);
    if (monitor !== undefined) {
      clearInterval(monitor);
      this.#segmentMonitors.delete(howl);
    }
    howl.off();
    howl.stop();
    howl.unload();
  }

  /**
   * A fragment from the trim bounds, or null to play the whole file. Invalid
   * bounds are ignored with a warning rather than breaking the scene's audio.
   * @returns {Segment | null}
   */
  static #toSegment(startTime, endTime, url) {
    const valid = (value) => value === null || (typeof value === 'number' && Number.isFinite(value) && value >= 0);
    if (!valid(startTime) || !valid(endTime)) {
      console.warn(`[AudioManager] Ignoring invalid trim for ${url}: start=${startTime}, end=${endTime}`);
      return null;
    }
    if (startTime === null && endTime === null) {
      return null;
    }
    const start = startTime ?? 0;
    if (endTime !== null && endTime <= start) {
      console.warn(`[AudioManager] Ignoring trim for ${url}: end ${endTime}s is not after start ${start}s`);
      return null;
    }
    return { start, end: endTime };
  }

  static #keyFor(url, segment) {
    return segment === null ? url : `${url}#t=${segment.start},${segment.end ?? ''}`;
  }
}

/** HTML5 MediaError codes, which Howler passes through for html5 sounds. */
const MEDIA_ERROR_MESSAGES = {
  1: 'loading was aborted',
  2: 'a network error occurred',
  3: 'the file could not be decoded',
  4: 'the file was not found or its format is not supported',
};

function describeHowlerError(error) {
  if (error === undefined || error === null) {
    return 'unknown error';
  }
  if (typeof error === 'number') {
    return MEDIA_ERROR_MESSAGES[error] ?? `media error ${error}`;
  }
  return typeof error === 'string' ? error : (error.message ?? String(error));
}

/** Shared audio manager for the whole game. */
export const audioManager = new AudioManager();
