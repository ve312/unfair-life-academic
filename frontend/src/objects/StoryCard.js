import Phaser from 'phaser';
import { COLORS, FONTS, TEXT_COLORS } from '../theme.js';
import { coverFit } from './imageFit.js';

const DEFAULTS = {
  width: 370,
  height: 230,
  coverHeight: 120,
  padding: 16,
};

const HOVER_SCALE = 1.03;
const PRESSED_SCALE = 0.98;
const SCALE_TWEEN_MS = 120;
const DESCRIPTION_LINES = 2;
const ELLIPSIS = '…';

/** Muted placeholder tints, picked per story so cards without a cover still differ. */
const PLACEHOLDER_TINTS = [0x2e3b55, 0x3b2e55, 0x2e554a, 0x55452e, 0x552e3b, 0x2e4d55];

/**
 * A selectable story in the story menu: a cover (image or placeholder), a
 * year badge, the title and a short description, grouped in a Container.
 *
 * The cover can be supplied later with setCover(textureKey) once it has been
 * loaded. Interaction mirrors ChoiceBtn: hover lifts the card, pressing sinks
 * it, and onSelect runs on pointerup only if the press started on the card.
 */
export class StoryCard extends Phaser.GameObjects.Container {
  /** @type {import('../api/bffClient.js').StorySummary} */
  story;
  #width;
  #coverHeight;
  /** @type {Phaser.GameObjects.Rectangle} */
  #background;
  /** @type {Phaser.GameObjects.Rectangle} */
  #placeholder;
  /** @type {Phaser.GameObjects.Text} */
  #placeholderInitial;
  /** @type {Phaser.GameObjects.Image | null} */
  #cover = null;
  /** @type {Phaser.GameObjects.Text} */
  #title;
  #onSelect;
  #enabled = true;
  #pressed = false;
  #hovered = false;
  /** @type {Phaser.Tweens.Tween | null} */
  #scaleTween = null;

  /**
   * @param {Phaser.Scene} scene
   * @param {number} x Centre x.
   * @param {number} y Centre y.
   * @param {import('../api/bffClient.js').StorySummary} story
   * @param {(story: import('../api/bffClient.js').StorySummary) => void} onSelect
   * @param {{ width?: number, height?: number, coverHeight?: number }} [options]
   */
  constructor(scene, x, y, story, onSelect, options = {}) {
    super(scene, x, y);
    if (typeof onSelect !== 'function') {
      throw new TypeError('StoryCard: onSelect must be a function');
    }
    const { width, height, coverHeight, padding } = { ...DEFAULTS, ...options };
    this.story = story;
    this.#width = width;
    this.#coverHeight = coverHeight;
    this.#onSelect = onSelect;

    const top = -height / 2;
    const coverCentreY = top + coverHeight / 2;
    const textWidth = width - padding * 2;

    this.#background = scene.add.rectangle(0, 0, width, height, COLORS.panel).setStrokeStyle(2, COLORS.border);

    this.#placeholder = scene.add.rectangle(0, coverCentreY, width - 4, coverHeight - 2, StoryCard.#tintFor(story.id));
    this.#placeholderInitial = scene.add
      .text(0, coverCentreY, story.title.trim().charAt(0).toUpperCase(), {
        fontFamily: FONTS.title,
        fontSize: '56px',
        color: TEXT_COLORS.muted,
      })
      .setOrigin(0.5)
      .setAlpha(0.6);

    const parts = [this.#background, this.#placeholder, this.#placeholderInitial];

    if (story.year !== null) {
      const badge = scene.add
        .text(width / 2 - 10, top + 10, String(story.year), {
          fontFamily: FONTS.body,
          fontSize: '15px',
          color: TEXT_COLORS.primary,
          backgroundColor: '#000000aa',
          padding: { x: 8, y: 3 },
        })
        .setOrigin(1, 0);
      parts.push(badge);
    }

    const textTop = top + coverHeight + 12;
    this.#title = scene.add
      .text(-width / 2 + padding, textTop, '', {
        fontFamily: FONTS.title,
        fontSize: '22px',
        color: TEXT_COLORS.primary,
        wordWrap: { width: textWidth, useAdvancedWrap: true },
      })
      .setOrigin(0, 0);
    StoryCard.#fitLines(this.#title, story.title, 1);

    const description = scene.add
      .text(-width / 2 + padding, textTop + 34, '', {
        fontFamily: FONTS.body,
        fontSize: '15px',
        color: TEXT_COLORS.muted,
        lineSpacing: 2,
        wordWrap: { width: textWidth, useAdvancedWrap: true },
      })
      .setOrigin(0, 0);
    StoryCard.#fitLines(description, story.shortDescription ?? 'No description yet.', DESCRIPTION_LINES);

    parts.push(this.#title, description);
    this.add(parts);

    this.setSize(width, height);
    this.setInteractive({ useHandCursor: true });
    this.on(Phaser.Input.Events.GAMEOBJECT_POINTER_OVER, this.#handleOver, this);
    this.on(Phaser.Input.Events.GAMEOBJECT_POINTER_OUT, this.#handleOut, this);
    this.on(Phaser.Input.Events.GAMEOBJECT_POINTER_DOWN, this.#handleDown, this);
    this.on(Phaser.Input.Events.GAMEOBJECT_POINTER_UP, this.#handleUp, this);
    scene.input.on(Phaser.Input.Events.POINTER_UP, this.#handleGlobalUp, this);
    this.once(Phaser.GameObjects.Events.DESTROY, this.#cleanup, this);

    scene.add.existing(this);
  }

  get enabled() {
    return this.#enabled;
  }

  get hasCover() {
    return this.#cover !== null;
  }

  /**
   * Replaces the placeholder with a loaded texture, cropped to fill the cover
   * area without distortion (like CSS background-size: cover).
   * @param {string} textureKey
   */
  setCover(textureKey) {
    if (!this.scene || !this.scene.textures.exists(textureKey)) {
      return this;
    }
    this.#cover?.destroy();
    const coverWidth = this.#width - 4;
    const coverHeight = this.#coverHeight - 2;
    const image = coverFit(this.scene.add.image(0, this.#placeholder.y, textureKey), coverWidth, coverHeight);

    // Above the placeholder, below the year badge and texts.
    this.addAt(image, this.getIndex(this.#placeholder) + 1);
    this.#placeholderInitial.setVisible(false);
    this.#cover = image;
    return this;
  }

  /** Disabled cards ignore input and fade out, e.g. while a story is starting. */
  setEnabled(enabled) {
    this.#enabled = Boolean(enabled);
    this.#pressed = false;
    if (this.#enabled) {
      this.setInteractive({ useHandCursor: true });
      this.setAlpha(1);
      this.#applyState(this.#hovered ? 'hover' : 'idle');
    } else {
      this.disableInteractive();
      this.#hovered = false;
      this.#applyState('idle');
    }
    return this;
  }

  /** Visual emphasis for the card the player picked while the others fade. */
  setHighlighted(highlighted) {
    this.#background.setStrokeStyle(highlighted ? 3 : 2, highlighted ? COLORS.accent : COLORS.border);
    this.setAlpha(highlighted ? 1 : 0.4);
    return this;
  }

  #handleOver() {
    this.#hovered = true;
    if (this.#enabled) {
      this.#applyState(this.#pressed ? 'pressed' : 'hover');
    }
  }

  #handleOut() {
    this.#hovered = false;
    if (this.#enabled) {
      this.#applyState('idle');
    }
  }

  #handleDown() {
    if (this.#enabled) {
      this.#pressed = true;
      this.#applyState('pressed');
    }
  }

  #handleUp() {
    if (!this.#enabled || !this.#pressed) {
      return;
    }
    this.#pressed = false;
    this.#applyState(this.#hovered ? 'hover' : 'idle');
    this.#onSelect(this.story);
  }

  #handleGlobalUp() {
    if (this.#pressed && !this.#hovered) {
      this.#pressed = false;
      if (this.#enabled) {
        this.#applyState('idle');
      }
    }
  }

  /** @param {'idle' | 'hover' | 'pressed'} state */
  #applyState(state) {
    const styles = {
      idle: { fill: COLORS.panel, stroke: COLORS.border, title: TEXT_COLORS.primary, scale: 1 },
      hover: { fill: COLORS.panelHover, stroke: COLORS.borderHover, title: TEXT_COLORS.accent, scale: HOVER_SCALE },
      pressed: { fill: COLORS.panelPressed, stroke: COLORS.borderHover, title: TEXT_COLORS.accent, scale: PRESSED_SCALE },
    };
    const style = styles[state];
    this.#background.setFillStyle(style.fill).setStrokeStyle(2, style.stroke);
    this.#title.setColor(style.title);
    if (state === 'hover') {
      // Draw the hovered card above its neighbours so its border is not overlapped.
      this.parentContainer?.bringToTop(this) ?? this.scene.children.bringToTop(this);
    }
    this.#scaleTween?.stop();
    this.#scaleTween = this.scene.tweens.add({ targets: this, scale: style.scale, duration: SCALE_TWEEN_MS, ease: 'Quad.easeOut' });
  }

  #cleanup() {
    this.#scaleTween?.stop();
    this.#scaleTween = null;
    this.scene?.input?.off(Phaser.Input.Events.POINTER_UP, this.#handleGlobalUp, this);
  }

  /**
   * Sets `text` so it wraps to at most `maxLines`, cutting at a word and
   * adding an ellipsis when it is longer.
   * @param {Phaser.GameObjects.Text} textObject
   * @param {string} text
   * @param {number} maxLines
   */
  static #fitLines(textObject, text, maxLines) {
    const clean = String(text).replace(/\s+/g, ' ').trim();
    if (textObject.getWrappedText(clean).length <= maxLines) {
      textObject.setText(clean);
      return;
    }
    const words = clean.split(' ');
    let low = 0;
    let high = words.length;
    // Longest word prefix that still fits once the ellipsis is added.
    while (low < high) {
      const mid = Math.ceil((low + high) / 2);
      if (textObject.getWrappedText(`${words.slice(0, mid).join(' ')}${ELLIPSIS}`).length <= maxLines) {
        low = mid;
      } else {
        high = mid - 1;
      }
    }
    textObject.setText(`${words.slice(0, Math.max(low, 1)).join(' ')}${ELLIPSIS}`);
  }

  static #tintFor(id) {
    let hash = 0;
    for (const char of String(id)) {
      hash = (hash * 31 + char.charCodeAt(0)) >>> 0;
    }
    return PLACEHOLDER_TINTS[hash % PLACEHOLDER_TINTS.length];
  }
}
