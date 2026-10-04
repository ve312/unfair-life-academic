/**
 * Read-only log of the current playthrough, kept only in the browser.
 *
 * It records the scenes the player has seen and what they chose, so the
 * History overlay can show the story so far. It never talks to the BFF and
 * offers no way to go back: past decisions cannot be changed.
 */

/**
 * @typedef {object} HistoryEntry
 * @property {number} step 1-based position in the playthrough.
 * @property {string} nodeId
 * @property {string} title
 * @property {string | null} backgroundMediaUrl
 * @property {number | null} year
 * @property {boolean} isCinematic
 * @property {{ text: string, requiresMobilityCheck: boolean, automatic: boolean } | null} choice
 *   What the player did to leave the scene; null for the scene being played.
 */

export class PlaythroughHistory {
  /** @type {{ id: string, title: string } | null} */
  #story = null;
  /** @type {HistoryEntry[]} */
  #entries = [];

  /** Starts a new log for a story, discarding the previous playthrough. */
  begin(story) {
    this.#story = story ? Object.freeze({ id: String(story.id), title: String(story.title) }) : null;
    this.#entries = [];
  }

  /** Forgets the current playthrough (e.g. when returning to the story hub). */
  clear() {
    this.begin(null);
  }

  get story() {
    return this.#story;
  }

  /**
   * Records that the player reached a scene. Re-entering the scene that is
   * already current (e.g. the scene restarted) is ignored.
   * @param {import('../api/bffClient.js').StoryNode} node
   */
  enter(node) {
    const current = this.#entries.at(-1);
    if (current && current.nodeId === node.id && current.choice === null) {
      return;
    }
    this.#entries.push({
      step: this.#entries.length + 1,
      nodeId: node.id,
      title: node.title,
      backgroundMediaUrl: node.backgroundMediaUrl ?? null,
      year: node.economy?.year ?? null,
      isCinematic: Boolean(node.isCinematic),
      choice: null,
    });
  }

  /**
   * Records how the player left the current scene.
   * @param {string} nodeId Must be the current scene.
   * @param {{ text: string, requiresMobilityCheck: boolean }} decision
   * @param {{ automatic?: boolean }} [options] automatic: the story moved on by itself (cinematic).
   */
  recordChoice(nodeId, decision, { automatic = false } = {}) {
    const current = this.#entries.at(-1);
    if (!current || current.nodeId !== nodeId) {
      return;
    }
    current.choice = {
      text: String(decision.text),
      requiresMobilityCheck: Boolean(decision.requiresMobilityCheck),
      automatic,
    };
  }

  /** Scenes the player has already left, oldest first, as frozen copies. */
  get passed() {
    return this.#entries
      .filter((entry) => entry.choice !== null)
      .map((entry) => Object.freeze({ ...entry, choice: Object.freeze({ ...entry.choice }) }));
  }
}

/** The single history of the running game. */
export const playthroughHistory = new PlaythroughHistory();
