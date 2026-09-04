/**
 * HTTP client for the Unfair Life BFF (NestJS).
 *
 * The client never sees where a decision leads: the BFF resolves the story
 * graph and returns only the next scene to render.
 */

const DEFAULT_BASE_URL = 'http://localhost:3000';
// 2 min: free-tier hosts (Render) sleep when idle and the first request wakes the
// BFF, which may in turn wake Strapi or the AI service (up to ~60 s each).
const DEFAULT_TIMEOUT_MS = 120000;
const INVALID_SCENE = 'The game server sent an invalid scene.';

/**
 * @typedef {object} AudioView
 * @property {string} url
 * @property {boolean} loop
 * @property {number} defaultVolume Between 0 and 1.
 * @property {number | null} [startTime] Seconds into the file where this scene's fragment starts.
 * @property {number | null} [endTime] Seconds into the file where the fragment ends (loops or stops).
 */

/**
 * @typedef {object} DecisionView
 * @property {string} id
 * @property {string} text
 * @property {boolean} requiresMobilityCheck True when the outcome depends on the AI prediction.
 */

/**
 * A playable story as listed by GET /game/stories.
 * @typedef {object} StorySummary
 * @property {string} id Pass it to startStory().
 * @property {string} title
 * @property {string | null} shortDescription
 * @property {string | null} coverImageUrl Absolute URL, or null when the story has no cover.
 * @property {number | null} year
 */

/**
 * A scene as returned by the BFF.
 * @typedef {object} StoryNode
 * @property {string} id
 * @property {string} title
 * @property {string | null} [context] Narrative context written in the CMS, shown under the title; null when none.
 * @property {string | null} backgroundMediaUrl
 * @property {number | null} [mediaStartTime] Fragment of a video background to loop, in seconds.
 * @property {number | null} [mediaEndTime]
 * @property {AudioView | null} audio
 * @property {number} timerDuration Seconds to choose; 0 means no timer.
 * @property {boolean} isCinematic
 * @property {boolean} isEnding
 * @property {EconomyView} [economy] What "mobility" means in the scene's year.
 * @property {DecisionView[]} decisions
 */

/**
 * @typedef {object} EconomyView
 * @property {number} year Year the scene takes place in.
 * @property {number} mobilityIncomeThreshold The model's "$50K in 1994" in this year's US dollars.
 * @property {boolean} cpiPublished False when the year's CPI is projected (2.5%/yr).
 */

/**
 * The 14 traits of the mobility model, using census spellings
 * (e.g. workclass 'Private', education 'HS-grad', nativeCountry 'Mexico').
 * @typedef {object} CharacterProfile
 * @property {number} age 17-90
 * @property {string} workclass
 * @property {string} education
 * @property {number} educationNum 1-16
 * @property {string} maritalStatus
 * @property {string} occupation
 * @property {string} relationship
 * @property {string} race
 * @property {string} sex
 * @property {number} hoursPerWeek 1-99
 * @property {string} nativeCountry
 * @property {number} [capitalGain]
 * @property {number} [capitalLoss]
 * @property {number} [fnlwgt]
 */

/**
 * Error raised for every failed BFF call.
 *
 * kind:
 *   'network' - the BFF could not be reached
 *   'timeout' - no answer within the timeout
 *   'aborted' - the caller cancelled the request (e.g. the scene changed)
 *   'http'    - the BFF answered with an error status (see status / details)
 *   'parse'   - the BFF answered with something that is not a valid scene
 */
export class BffError extends Error {
  /**
   * @param {string} message Human-readable, safe to show to the player.
   * @param {{ kind: 'network' | 'timeout' | 'aborted' | 'http' | 'parse', status?: number, details?: string[], cause?: unknown }} info
   */
  constructor(message, { kind, status = null, details = [], cause } = {}) {
    super(message, { cause });
    this.name = 'BffError';
    this.kind = kind;
    /** @type {number | null} */
    this.status = status;
    /** Field-level validation messages returned by the BFF (400 responses). */
    this.details = details;
  }

  /** True when trying the same request again may succeed. */
  get retryable() {
    return (
      this.kind === 'network' ||
      this.kind === 'timeout' ||
      this.status === 502 ||
      this.status === 503 ||
      this.status === 504
    );
  }
}

export class BffClient {
  /**
   * @param {{ baseUrl?: string, timeoutMs?: number, fetchImpl?: typeof fetch }} [options]
   */
  constructor({ baseUrl = resolveDefaultBaseUrl(), timeoutMs = DEFAULT_TIMEOUT_MS, fetchImpl } = {}) {
    let parsed;
    try {
      parsed = new URL(baseUrl);
    } catch {
      throw new TypeError(`BffClient: baseUrl must be an absolute URL, got '${baseUrl}'`);
    }
    if (parsed.protocol !== 'http:' && parsed.protocol !== 'https:') {
      throw new TypeError(`BffClient: baseUrl must use http or https, got '${baseUrl}'`);
    }
    if (!Number.isInteger(timeoutMs) || timeoutMs <= 0) {
      throw new TypeError(`BffClient: timeoutMs must be a positive integer, got '${timeoutMs}'`);
    }
    this.baseUrl = baseUrl.replace(/\/+$/, '');
    this.timeoutMs = timeoutMs;
    this.fetchImpl = fetchImpl ?? globalThis.fetch.bind(globalThis);
  }

  /**
   * Lists the stories the player can start (only stories with a starting scene).
   * @param {{ signal?: AbortSignal }} [options]
   * @returns {Promise<StorySummary[]>}
   */
  async getStories({ signal } = {}) {
    return this.#request('GET', '/game/stories', undefined, signal, isStoryList, 'The game server sent an invalid story list.');
  }

  /**
   * Starts a story and returns its first scene.
   * @param {string} storyId Strapi documentId of the story.
   * @param {{ signal?: AbortSignal }} [options]
   * @returns {Promise<StoryNode>}
   */
  async startStory(storyId, { signal } = {}) {
    requireId('storyId', storyId);
    return this.#request('POST', `/game/start/${encodeURIComponent(storyId)}`, undefined, signal, isStoryNode, INVALID_SCENE);
  }

  /**
   * Submits the player's choice and returns the scene it leads to.
   * Mobility checks run on the story's base profile, defined in the CMS, so the
   * game passes no character. characterProfile remains only as a legacy
   * fallback for stories without a base_profile; the BFF ignores it otherwise.
   * @param {string} currentNodeId
   * @param {string} decisionId
   * @param {CharacterProfile | null} [characterProfile]
   * @param {{ signal?: AbortSignal }} [options]
   * @returns {Promise<StoryNode>}
   */
  async submitDecision(currentNodeId, decisionId, characterProfile = null, { signal } = {}) {
    requireId('currentNodeId', currentNodeId);
    requireId('decisionId', decisionId);
    if (characterProfile !== null && (typeof characterProfile !== 'object' || Array.isArray(characterProfile))) {
      throw new TypeError('submitDecision: characterProfile must be an object or null');
    }
    const body = { currentNodeId, decisionId };
    if (characterProfile !== null) {
      body.character = characterProfile;
    }
    return this.#request('POST', '/game/decision', body, signal, isStoryNode, INVALID_SCENE);
  }

  /**
   * Sends a request and returns the parsed body once `isValid` accepts it.
   * @template T
   * @param {'GET' | 'POST'} method
   * @param {string} path
   * @param {unknown} body
   * @param {AbortSignal | undefined} callerSignal
   * @param {(payload: unknown) => payload is T} isValid
   * @param {string} invalidMessage
   * @returns {Promise<T>}
   */
  async #request(method, path, body, callerSignal, isValid, invalidMessage) {
    const timeoutSignal = AbortSignal.timeout(this.timeoutMs);
    const signal = callerSignal ? combineSignals(callerSignal, timeoutSignal) : timeoutSignal;

    let response;
    try {
      response = await this.fetchImpl(`${this.baseUrl}${path}`, {
        method,
        headers: {
          Accept: 'application/json',
          ...(body === undefined ? {} : { 'Content-Type': 'application/json' }),
        },
        body: body === undefined ? undefined : JSON.stringify(body),
        signal,
      });
    } catch (error) {
      if (callerSignal?.aborted) {
        throw new BffError('The request was cancelled.', { kind: 'aborted', cause: error });
      }
      if (timeoutSignal.aborted) {
        throw new BffError('The game server took too long to respond. Please try again.', {
          kind: 'timeout',
          cause: error,
        });
      }
      throw new BffError('Cannot reach the game server. Check your connection and try again.', {
        kind: 'network',
        cause: error,
      });
    }

    const payload = await readJson(response);

    if (!response.ok) {
      const { message, details } = describeErrorPayload(payload, response.status);
      throw new BffError(message, { kind: 'http', status: response.status, details });
    }
    if (!isValid(payload)) {
      throw new BffError(invalidMessage, { kind: 'parse', status: response.status });
    }
    return payload;
  }
}

/** Shared client configured from VITE_BFF_URL (or http://localhost:3000). */
export const bffClient = new BffClient();

function resolveDefaultBaseUrl() {
  const fromEnv = import.meta.env?.VITE_BFF_URL;
  return typeof fromEnv === 'string' && fromEnv.trim() !== '' ? fromEnv.trim() : DEFAULT_BASE_URL;
}

function requireId(name, value) {
  if (typeof value !== 'string' || value.trim() === '') {
    throw new TypeError(`${name} must be a non-empty string`);
  }
}

/**
 * Aborts when either signal aborts. Uses AbortSignal.any where available.
 * @param {AbortSignal} first
 * @param {AbortSignal} second
 */
function combineSignals(first, second) {
  if (typeof AbortSignal.any === 'function') {
    return AbortSignal.any([first, second]);
  }
  const controller = new AbortController();
  for (const signal of [first, second]) {
    if (signal.aborted) {
      controller.abort(signal.reason);
      break;
    }
    signal.addEventListener('abort', () => controller.abort(signal.reason), { once: true });
  }
  return controller.signal;
}

/**
 * Reads a JSON body; returns null for an empty body and throws a parse error
 * for anything that is not JSON (e.g. an HTML error page from a proxy).
 * @param {Response} response
 */
async function readJson(response) {
  let text;
  try {
    text = await response.text();
  } catch (error) {
    throw new BffError('The connection to the game server was interrupted.', {
      kind: 'network',
      status: response.status,
      cause: error,
    });
  }
  if (text.trim() === '') {
    return null;
  }
  try {
    return JSON.parse(text);
  } catch (error) {
    throw new BffError(`The game server sent an unreadable response (HTTP ${response.status}).`, {
      kind: 'parse',
      status: response.status,
      cause: error,
    });
  }
}

/**
 * Extracts a message from a NestJS error body: { statusCode, error, message }
 * where message is a string, or an array of field errors for 400 responses.
 * @param {unknown} payload
 * @param {number} status
 */
function describeErrorPayload(payload, status) {
  const message = payload && typeof payload === 'object' ? payload.message : undefined;
  if (Array.isArray(message)) {
    const details = message.filter((item) => typeof item === 'string');
    return { message: 'The game server rejected the request.', details };
  }
  if (typeof message === 'string' && message.trim() !== '') {
    return { message, details: [] };
  }
  return { message: `The game server returned an error (HTTP ${status}).`, details: [] };
}

/** @returns {payload is StoryNode} */
function isStoryNode(payload) {
  if (!payload || typeof payload !== 'object') {
    return false;
  }
  const audioOk =
    payload.audio === null ||
    (typeof payload.audio === 'object' &&
      typeof payload.audio.url === 'string' &&
      typeof payload.audio.loop === 'boolean' &&
      typeof payload.audio.defaultVolume === 'number');
  return (
    typeof payload.id === 'string' &&
    typeof payload.title === 'string' &&
    // Optional, so the client keeps working against a BFF deployed before the field existed.
    (payload.context === undefined || payload.context === null || typeof payload.context === 'string') &&
    (payload.backgroundMediaUrl === null || typeof payload.backgroundMediaUrl === 'string') &&
    audioOk &&
    Number.isInteger(payload.timerDuration) &&
    typeof payload.isCinematic === 'boolean' &&
    typeof payload.isEnding === 'boolean' &&
    Array.isArray(payload.decisions) &&
    payload.decisions.every(
      (decision) =>
        decision &&
        typeof decision.id === 'string' &&
        typeof decision.text === 'string' &&
        typeof decision.requiresMobilityCheck === 'boolean',
    )
  );
}

/** @returns {payload is StorySummary[]} */
function isStoryList(payload) {
  return (
    Array.isArray(payload) &&
    payload.every(
      (story) =>
        story !== null &&
        typeof story === 'object' &&
        typeof story.id === 'string' &&
        story.id !== '' &&
        typeof story.title === 'string' &&
        (story.shortDescription === null || typeof story.shortDescription === 'string') &&
        (story.coverImageUrl === null || typeof story.coverImageUrl === 'string') &&
        (story.year === null || Number.isInteger(story.year)),
    )
  );
}
