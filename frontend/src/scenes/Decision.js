import Phaser from 'phaser';
import { bffClient } from '../api/bffClient.js';
import { audioManager } from '../managers/audio.js';
import { playthroughHistory } from '../managers/history.js';
import { ChoiceBtn } from '../objects/ChoiceBtn.js';
import { COLORS, FONTS, GAME_HEIGHT, GAME_WIDTH, TEXT_COLORS } from '../theme.js';

const BUTTON_WIDTH = 760;
const BUTTON_HEIGHT = 72;
const BUTTON_GAP = 18;
const BUTTON_AREA_TOP = GAME_HEIGHT * 0.42;
const BACKGROUND_DIM_ALPHA = 0.55;
const FADE_IN_MS = 350;
const FADE_OUT_MS = 250;
/** How long the scene's UI takes to fade in once its timer has elapsed. */
const UI_REVEAL_MS = 600;
/** How long a cinematic scene plays when the CMS gives it no timer of its own. */
const CINEMATIC_HOLD_MS = 4000;
/** Narrative context under the title: tried at these sizes (px) until it fits above the choices. */
const CONTEXT_FONT_SIZES = [20, 18, 16];
const CONTEXT_LINE_SPACING = 6;
const CONTEXT_GAP = 18;
/** Lowest point the context may reach in a cinematic scene, which has no choice buttons. */
const CINEMATIC_CONTEXT_BOTTOM = GAME_HEIGHT * 0.88;
/** Lowest point the context may reach in an ending, which shows "The End" at GAME_HEIGHT * 0.45. */
const ENDING_CONTEXT_BOTTOM = GAME_HEIGHT * 0.45 - 34;
const VIDEO_EXTENSION = /\.(mp4|webm|ogv|mov)(\?.*)?$/i;
/** A trimmed video loops back this long before the file's end, before the browser's own loop restarts it at 0. */
const VIDEO_LOOP_MARGIN_S = 0.05;

/**
 * The game loop: renders one story scene (a StoryNode from the BFF), lets the
 * player choose, submits the choice and restarts itself with the next scene.
 *
 * timer_duration controls pacing, not a countdown:
 *   - choice scene: background, title and audio play for the full timer;
 *     only then do the decision buttons fade in and become clickable. The
 *     scene then waits for the player as long as needed (no auto-skip).
 *   - ending (no decisions): same reveal for "The End" and the return button.
 *   - cinematic: plays for its full timer (CINEMATIC_HOLD_MS when it has none)
 *     with NO way to skip, then continues through its first decision.
 *   - timer 0: the UI appears immediately.
 *
 * A read-only History overlay (scene 'History') can be opened from any
 * non-cinematic scene; it pauses this scene, so a pending reveal waits too.
 * Listeners are removed on SHUTDOWN, so restarts never stack handlers.
 */
export class Decision extends Phaser.Scene {
  static KEY = 'Decision';

  static EVENTS = Object.freeze({
    CHOICE: 'decision:choice',
  });

  /** @type {import('../api/bffClient.js').StoryNode | null} */
  node = null;
  /** @type {ChoiceBtn[]} */
  buttons = [];
  /** True once the scene's UI is visible and clickable (or, for cinematics, once it has moved on). */
  uiRevealed = false;
  #resolved = false;
  /** @type {AbortController | null} */
  #request = null;
  /** @type {Phaser.GameObjects.Container | null} */
  #statusPanel = null;
  /** @type {Phaser.Time.TimerEvent | null} */
  #revealTimer = null;
  /** @type {ChoiceBtn | null} */
  #historyButton = null;
  /** @type {{ video: Phaser.GameObjects.Video, start: number, end: number } | null} */
  #videoFragment = null;
  /** Finishes a video trim once the file's duration is known; returns true when done. */
  #pendingVideoMarkers = null;

  constructor() {
    super(Decision.KEY);
  }

  /** @param {{ node: import('../api/bffClient.js').StoryNode }} data */
  init(data) {
    if (!data?.node || typeof data.node.id !== 'string' || !Array.isArray(data.node.decisions)) {
      throw new TypeError("Decision scene requires { node } with an 'id' and a 'decisions' array");
    }
    this.node = data.node;
    this.buttons = [];
    this.uiRevealed = false;
    this.#resolved = false;
    this.#request = null;
    this.#statusPanel = null;
    this.#revealTimer = null;
    this.#historyButton = null;
    this.#videoFragment = null;
    this.#pendingVideoMarkers = null;
  }

  create() {
    this.cameras.main.setBackgroundColor(COLORS.background);

    this.events.on(Decision.EVENTS.CHOICE, this.#handleChoice, this);
    this.events.once(Phaser.Scenes.Events.SHUTDOWN, this.#onShutdown, this);
    playthroughHistory.enter(this.node);

    // Scenes without their own track keep the current music playing.
    if (this.node.audio) {
      const { url, loop, defaultVolume, startTime = null, endTime = null } = this.node.audio;
      audioManager.playTrack(url, loop, defaultVolume, { startTime, endTime });
    }

    this.#loadBackground().then((background) => {
      if (this.sys.isActive()) {
        this.#render(background);
      }
    });
  }

  /**
   * Keeps a trimmed background video inside its fragment. Phaser's own marker
   * check runs in the browser's video-frame callback, which only fires while
   * frames are being composited; this per-step check makes the loop independent
   * of that. Both jumping back at the same moment is harmless.
   */
  update() {
    if (this.#pendingVideoMarkers?.()) {
      this.#pendingVideoMarkers = null;
    }
    const fragment = this.#videoFragment;
    const element = fragment?.video.video;
    if (element && !element.paused && element.currentTime >= fragment.end) {
      element.currentTime = fragment.start;
    }
  }

  /** Milliseconds the scene plays before its UI appears (or, for cinematics, before it moves on). */
  get revealDelayMs() {
    const timerMs = Math.max(0, this.node.timerDuration) * 1000;
    if (this.node.isCinematic && !this.node.isEnding) {
      return timerMs > 0 ? timerMs : CINEMATIC_HOLD_MS;
    }
    return timerMs;
  }

  // ---------------------------------------------------------------- loading

  /**
   * Loads the scene's background through Phaser's Loader and resolves once the
   * loader reports 'complete'. Resolves null when there is no background or it
   * failed to load, so a broken media URL never blocks the story.
   * @returns {Promise<{ key: string, isVideo: boolean } | null>}
   */
  #loadBackground() {
    const url = this.node.backgroundMediaUrl;
    if (!url) {
      return Promise.resolve(null);
    }
    const isVideo = VIDEO_EXTENSION.test(url);
    const key = `background:${url}`;
    if (!isVideo && this.textures.exists(key)) {
      return Promise.resolve({ key, isVideo });
    }
    if (isVideo && this.cache.video.exists(key)) {
      return Promise.resolve({ key, isVideo });
    }

    const loading = this.add
      .text(GAME_WIDTH / 2, GAME_HEIGHT / 2, 'Loading...', { fontFamily: FONTS.body, fontSize: '22px', color: TEXT_COLORS.muted })
      .setOrigin(0.5);

    return new Promise((resolve) => {
      let failed = false;
      const onError = (file) => {
        if (file.key === key) {
          failed = true;
          console.error(`[Decision] Background failed to load: ${url}`);
        }
      };
      this.load.on(Phaser.Loader.Events.FILE_LOAD_ERROR, onError);
      this.load.once(Phaser.Loader.Events.COMPLETE, () => {
        this.load.off(Phaser.Loader.Events.FILE_LOAD_ERROR, onError);
        loading.destroy();
        resolve(failed ? null : { key, isVideo });
      });
      if (isVideo) {
        this.load.video(key, url, true);
      } else {
        this.load.image(key, url);
      }
      this.load.start();
    });
  }

  // -------------------------------------------------------------- rendering

  /** @param {{ key: string, isVideo: boolean } | null} background */
  #render(background) {
    const { node } = this;

    if (background) {
      this.#addBackground(background);
    }

    const title = this.add
      .text(GAME_WIDTH / 2, GAME_HEIGHT * 0.16, node.title, {
        fontFamily: FONTS.title,
        fontSize: '52px',
        color: TEXT_COLORS.primary,
        align: 'center',
        wordWrap: { width: GAME_WIDTH - 160, useAdvancedWrap: true },
      })
      .setOrigin(0.5)
      .setShadow(0, 4, '#000000', 10, false, true);
    this.#addContext(title.getBounds().bottom + CONTEXT_GAP);

    this.cameras.main.fadeIn(FADE_IN_MS, 0, 0, 0);

    if (node.isCinematic && !node.isEnding) {
      // Mandatory viewing: no skip input and no History button while it plays.
      this.#revealTimer = this.time.delayedCall(this.revealDelayMs, () => this.#continueCinematic());
      return;
    }

    const ui = node.isEnding ? this.#buildEnding() : this.#buildChoices();
    const delay = this.revealDelayMs;
    if (delay === 0) {
      this.#addHistoryButton();
      this.uiRevealed = true;
      return;
    }
    // Hidden and inert from the very first frame; only the media, the title and the audio play.
    // The History button does not exist yet: it arrives with the reveal.
    ui.objects.forEach((object) => object.setAlpha(0));
    ui.buttons.forEach((button) => button.setEnabled(false));
    this.#revealTimer = this.time.delayedCall(delay, () => this.#revealUi(ui));
  }

  /**
   * Shows the scene's narrative context (written in the CMS) under the title, with it,
   * from the first frame. The text must never overlap the choices that fade in later, so
   * it is fitted into the space above them: first by stepping the font down, then, if it
   * is still too long, by cutting it after the last line that fits and adding an ellipsis.
   * @param {number} top y where the context starts (just below the title).
   */
  #addContext(top) {
    const context = this.node.context;
    if (typeof context !== 'string' || context.trim() === '') {
      return;
    }
    const { isCinematic, isEnding } = this.node;
    const bottom = isEnding ? ENDING_CONTEXT_BOTTOM : isCinematic ? CINEMATIC_CONTEXT_BOTTOM : BUTTON_AREA_TOP - CONTEXT_GAP;
    const maxHeight = bottom - top;
    if (maxHeight < CONTEXT_FONT_SIZES.at(-1) * 2) {
      return; // A title that wraps onto many lines leaves no room worth using.
    }

    const text = this.add
      .text(GAME_WIDTH / 2, top, '', {
        fontFamily: FONTS.body,
        color: TEXT_COLORS.primary,
        align: 'center',
        lineSpacing: CONTEXT_LINE_SPACING,
        wordWrap: { width: BUTTON_WIDTH + 80, useAdvancedWrap: true },
      })
      .setOrigin(0.5, 0)
      .setShadow(0, 2, '#000000', 6, false, true);

    for (const size of CONTEXT_FONT_SIZES) {
      text.setFontSize(size).setText(context.trim());
      if (text.height <= maxHeight) {
        return;
      }
    }
    // Still too tall at the smallest size: keep the lines that fit and mark the cut.
    const lines = text.getWrappedText(context.trim());
    const lineHeight = text.height / Math.max(lines.length, 1);
    // The ellipsis can push the last line onto a new one, so drop lines until the cut fits.
    for (let keep = Math.max(1, Math.floor(maxHeight / lineHeight)); keep >= 1; keep--) {
      text.setText(`${lines.slice(0, keep).join('\n').replace(/[\s.,;:]*$/, '')}…`);
      if (text.height <= maxHeight) {
        return;
      }
    }
  }

  #addBackground({ key, isVideo }) {
    const media = isVideo
      ? this.#playBackgroundVideo(this.add.video(GAME_WIDTH / 2, GAME_HEIGHT / 2, key).setMute(true))
      : this.add.image(GAME_WIDTH / 2, GAME_HEIGHT / 2, key);

    // Cover the whole 1280x720 frame, cropping overflow, like CSS background-size: cover.
    const fitCover = () => {
      const width = media.width || GAME_WIDTH;
      const height = media.height || GAME_HEIGHT;
      media.setScale(Math.max(GAME_WIDTH / width, GAME_HEIGHT / height));
    };
    fitCover();
    if (isVideo) {
      // Video dimensions are only known once metadata has loaded.
      media.once(Phaser.GameObjects.Events.VIDEO_CREATED, fitCover);
    }

    this.add.rectangle(GAME_WIDTH / 2, GAME_HEIGHT / 2, GAME_WIDTH, GAME_HEIGHT, 0x000000, BACKGROUND_DIM_ALPHA);
  }

  /**
   * Starts a background video, looping only the scene's fragment when one is set
   * (mediaStartTime / mediaEndTime, in seconds). Phaser's marker playback seeks to
   * the start and jumps back from the end; without an end time the fragment runs
   * to just before the file's end, so the loop returns to the start instead of 0.
   * @param {Phaser.GameObjects.Video} video
   */
  #playBackgroundVideo(video) {
    const start = this.node.mediaStartTime ?? null;
    const end = this.node.mediaEndTime ?? null;
    if (start === null && end === null) {
      return video.play(true);
    }
    const markerIn = start ?? 0;

    // The duration is NaN until the file's metadata arrives (large files stream in),
    // so the final markers are set once it is known. Calling play() again on a
    // playing video only updates its markers.
    const applyMarkers = () => {
      const duration = video.getDuration();
      if (!Number.isFinite(duration) || duration <= 0) {
        return false;
      }
      if (markerIn >= duration - VIDEO_LOOP_MARGIN_S) {
        console.warn(`[Decision] Video trim starts after the end of the file (${markerIn}s >= ${duration}s); playing it whole`);
        video.play(true);
        return true;
      }
      const markerOut = Math.min(end ?? duration, duration - VIDEO_LOOP_MARGIN_S);
      video.play(true, markerIn, markerOut);
      this.#videoFragment = { video, start: markerIn, end: markerOut };
      return true;
    };

    // Start right away at the fragment (Phaser seeks when playback begins), then refine.
    video.play(true, markerIn, end ?? undefined);
    if (end !== null) {
      this.#videoFragment = { video, start: markerIn, end };
    }
    if (!applyMarkers()) {
      // Polled from update(): a cached video may never emit another metadata event.
      this.#pendingVideoMarkers = applyMarkers;
    }
    return video;
  }

  /** Decision buttons, created hidden and inert until the reveal. */
  #buildChoices() {
    const centerX = GAME_WIDTH / 2;
    const { decisions } = this.node;
    const stackHeight = decisions.length * BUTTON_HEIGHT + (decisions.length - 1) * BUTTON_GAP;
    const firstY = Math.max(BUTTON_AREA_TOP, (GAME_HEIGHT - stackHeight) / 2 + 60) + BUTTON_HEIGHT / 2;

    this.buttons = decisions.map((decision, index) => {
      const label = decision.text;
      return new ChoiceBtn(
        this,
        centerX,
        firstY + index * (BUTTON_HEIGHT + BUTTON_GAP),
        label,
        () => this.events.emit(Decision.EVENTS.CHOICE, decision),
        { width: BUTTON_WIDTH, height: BUTTON_HEIGHT },
      );
    });
    return { objects: [...this.buttons], buttons: [...this.buttons] };
  }

  /** "The End" and the way back to the story hub, created hidden and inert until the reveal. */
  #buildEnding() {
    const theEnd = this.add
      .text(GAME_WIDTH / 2, GAME_HEIGHT * 0.45, 'The End', { fontFamily: FONTS.title, fontSize: '40px', color: TEXT_COLORS.accent })
      .setOrigin(0.5);
    const back = new ChoiceBtn(this, GAME_WIDTH / 2, GAME_HEIGHT * 0.66, 'Back to Stories', () => this.#returnToStories(), {
      width: 420,
      height: 72,
    });
    return { objects: [theEnd, back], buttons: [back] };
  }

  /** Fades the scene's UI in once its timer has run out; buttons become clickable when fully visible. */
  #revealUi({ objects, buttons }) {
    this.#revealTimer = null;
    if (this.#resolved) {
      return;
    }
    const historyButton = this.#addHistoryButton();
    if (historyButton) {
      historyButton.setAlpha(0);
      objects = [...objects, historyButton];
      buttons = [...buttons, historyButton];
    }
    // Normal colours while fading in, but no input until fully visible.
    buttons.forEach((button) => button.setEnabled(true).disableInteractive());
    this.tweens.add({
      targets: objects,
      alpha: 1,
      duration: UI_REVEAL_MS,
      ease: 'Sine.easeOut',
      onComplete: () => {
        if (!this.#resolved) {
          buttons.forEach((button) => button.setInteractive({ useHandCursor: true }));
          this.uiRevealed = true;
        }
      },
    });
  }

  /** The cinematic has played in full: the story moves on through its first decision. */
  #continueCinematic() {
    this.#revealTimer = null;
    this.uiRevealed = true;
    const [next] = this.node.decisions;
    this.events.emit(Decision.EVENTS.CHOICE, next, { automatic: true });
  }

  /** Adds the History button when there is something to review; returns it, or null. */
  #addHistoryButton() {
    if (playthroughHistory.passed.length === 0) {
      return null;
    }
    this.#historyButton = new ChoiceBtn(this, GAME_WIDTH - 100, 40, 'History', () => this.#openHistory(), {
      width: 150,
      height: 46,
      fontSize: 18,
    }).setDepth(5);
    return this.#historyButton;
  }

  #openHistory() {
    if (this.#resolved) {
      return;
    }
    // Pausing freezes this scene's clock and tweens, so a pending reveal waits for the player.
    this.scene.launch('History', { returnTo: Decision.KEY });
    this.scene.pause();
  }

  // -------------------------------------------------------------- game loop

  /**
   * @param {import('../api/bffClient.js').DecisionView} decision
   * @param {{ automatic?: boolean }} [options]
   */
  #handleChoice(decision, { automatic = false } = {}) {
    if (this.#resolved || !decision) {
      return;
    }
    this.#lockInput(decision);
    playthroughHistory.recordChoice(this.node.id, decision, { automatic });
    this.#submit(decision);
  }

  #lockInput(chosen) {
    this.#resolved = true;
    this.#revealTimer?.remove(false);
    this.#revealTimer = null;
    this.#historyButton?.setEnabled(false);
    this.buttons.forEach((button, index) => {
      button.setEnabled(false);
      // Keep the chosen option readable so the player sees what was submitted.
      button.setAlpha(this.node.decisions[index] === chosen ? 1 : 0.45);
    });
  }

  /** @param {import('../api/bffClient.js').DecisionView} decision */
  async #submit(decision) {
    if (!this.#statusPanel) {
      this.#showStatus(decision.requiresMobilityCheck ? 'Life is weighing your chances...' : '...');
    }
    this.#request = new AbortController();

    try {
      // No character is sent: the BFF evaluates the story's base profile (CMS) plus the decision's hidden modifiers.
      const next = await bffClient.submitDecision(this.node.id, decision.id, null, { signal: this.#request.signal });
      this.#request = null;
      if (!this.sys.isActive()) {
        return;
      }
      this.cameras.main.fadeOut(FADE_OUT_MS, 0, 0, 0);
      this.cameras.main.once(Phaser.Cameras.Scene2D.Events.FADE_OUT_COMPLETE, () => this.scene.restart({ node: next }));
    } catch (error) {
      this.#request = null;
      if (error?.kind === 'aborted' || !this.sys.isActive()) {
        return;
      }
      console.error('[Decision] Could not submit the decision:', error);
      this.#showError(error?.message ?? 'Something went wrong.', decision);
    }
  }

  #returnToStories() {
    audioManager.stopCurrentTrack();
    playthroughHistory.clear();
    this.scene.start('StorySelection');
  }

  // ------------------------------------------------------------ status panel

  /** Dims the screen and shows a pulsing message while the BFF answers. */
  #showStatus(message) {
    this.#clearStatus();
    const overlay = this.add.rectangle(0, 0, GAME_WIDTH, GAME_HEIGHT, 0x000000, 0.45).setOrigin(0);
    const text = this.add
      .text(GAME_WIDTH / 2, GAME_HEIGHT - 70, message, { fontFamily: FONTS.body, fontSize: '24px', color: TEXT_COLORS.accent })
      .setOrigin(0.5);
    this.tweens.add({ targets: text, alpha: 0.35, duration: 700, yoyo: true, repeat: -1, ease: 'Sine.easeInOut' });
    this.#statusPanel = this.add.container(0, 0, [overlay, text]).setDepth(10);
  }

  /** Shows a failed request with a retry for the same decision and a way out. */
  #showError(message, decision) {
    this.#clearStatus();
    const overlay = this.add.rectangle(0, 0, GAME_WIDTH, GAME_HEIGHT, 0x000000, 0.7).setOrigin(0);
    const text = this.add
      .text(GAME_WIDTH / 2, GAME_HEIGHT * 0.4, message, {
        fontFamily: FONTS.body,
        fontSize: '24px',
        color: '#e07a6a',
        align: 'center',
        wordWrap: { width: GAME_WIDTH - 240 },
      })
      .setOrigin(0.5);
    const retry = new ChoiceBtn(this, GAME_WIDTH / 2, GAME_HEIGHT * 0.58, 'Try Again', () => {
      this.#showStatus('...');
      this.#submit(decision);
    }, { width: 360, height: 68 });
    const back = new ChoiceBtn(this, GAME_WIDTH / 2, GAME_HEIGHT * 0.58 + 88, 'Back to Stories', () => this.#returnToStories(), {
      width: 360,
      height: 68,
    });
    this.#statusPanel = this.add.container(0, 0, [overlay, text, retry, back]).setDepth(10);
  }

  #clearStatus() {
    this.#statusPanel?.destroy();
    this.#statusPanel = null;
  }

  // ---------------------------------------------------------------- cleanup

  #onShutdown() {
    this.events.off(Decision.EVENTS.CHOICE, this.#handleChoice, this);
    this.#request?.abort();
    this.#request = null;
    this.#revealTimer?.remove(false);
    this.#revealTimer = null;
  }
}
