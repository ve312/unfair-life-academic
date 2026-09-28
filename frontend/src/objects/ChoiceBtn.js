import Phaser from 'phaser';
import { COLORS, FONTS, TEXT_COLORS } from '../theme.js';

const DEFAULTS = {
  width: 640,
  height: 72,
  fontSize: 24,
  padding: 24,
};

const HOVER_SCALE = 1.04;
const PRESSED_SCALE = 0.97;
const SCALE_TWEEN_MS = 120;

/**
 * A clickable choice button: a rounded-looking panel (Rectangle) with a
 * centred, word-wrapped label (Text), grouped in a Container.
 *
 * Visual feedback: hover lightens the panel, highlights the border and grows
 * the button slightly; pressing darkens and shrinks it. The callback runs on
 * pointerup only if the press started on the button and the pointer is still
 * over it, matching how native buttons behave.
 */
export class ChoiceBtn extends Phaser.GameObjects.Container {
  /** @type {Phaser.GameObjects.Rectangle} */
  #background;
  /** @type {Phaser.GameObjects.Text} */
  #label;
  /** @type {() => void} */
  #onClick;
  #enabled = true;
  #pressed = false;
  #hovered = false;
  /** @type {Phaser.Tweens.Tween | null} */
  #scaleTween = null;

  /**
   * @param {Phaser.Scene} scene
   * @param {number} x Centre x.
   * @param {number} y Centre y.
   * @param {string} text Label shown on the button.
   * @param {() => void} onClick Called when the button is clicked.
   * @param {{ width?: number, height?: number, fontSize?: number }} [options]
   */
  constructor(scene, x, y, text, onClick, options = {}) {
    super(scene, x, y);
    if (typeof onClick !== 'function') {
      throw new TypeError('ChoiceBtn: onClick must be a function');
    }
    const { width, height, fontSize, padding } = { ...DEFAULTS, ...options };
    this.#onClick = onClick;

    this.#background = scene.add
      .rectangle(0, 0, width, height, COLORS.panel)
      .setStrokeStyle(2, COLORS.border);

    this.#label = scene.add
      .text(0, 0, String(text), {
        fontFamily: FONTS.body,
        fontSize: `${fontSize}px`,
        color: TEXT_COLORS.primary,
        align: 'center',
        wordWrap: { width: width - padding * 2, useAdvancedWrap: true },
      })
      .setOrigin(0.5);

    this.add([this.#background, this.#label]);
    this.setSize(width, height);
    this.setInteractive({ useHandCursor: true });

    this.on(Phaser.Input.Events.GAMEOBJECT_POINTER_OVER, this.#handleOver, this);
    this.on(Phaser.Input.Events.GAMEOBJECT_POINTER_OUT, this.#handleOut, this);
    this.on(Phaser.Input.Events.GAMEOBJECT_POINTER_DOWN, this.#handleDown, this);
    this.on(Phaser.Input.Events.GAMEOBJECT_POINTER_UP, this.#handleUp, this);
    // A press that ends outside the button must not leave it stuck in the pressed state.
    scene.input.on(Phaser.Input.Events.POINTER_UP, this.#handleGlobalUp, this);
    this.once(Phaser.GameObjects.Events.DESTROY, this.#cleanup, this);

    scene.add.existing(this);
  }

  get enabled() {
    return this.#enabled;
  }

  /** Changes the label text. */
  setText(text) {
    this.#label.setText(String(text));
    return this;
  }

  /**
   * Enables or disables the button. Disabled buttons ignore input and are greyed out,
   * e.g. right after a choice is made so it cannot be submitted twice.
   * @param {boolean} enabled
   */
  setEnabled(enabled) {
    this.#enabled = Boolean(enabled);
    this.#pressed = false;
    if (this.#enabled) {
      this.setInteractive({ useHandCursor: true });
      this.#applyState(this.#hovered ? 'hover' : 'idle');
    } else {
      this.disableInteractive();
      this.#hovered = false;
      this.#applyState('disabled');
    }
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
    if (!this.#enabled) {
      return;
    }
    this.#pressed = true;
    this.#applyState('pressed');
  }

  #handleUp() {
    if (!this.#enabled || !this.#pressed) {
      return;
    }
    this.#pressed = false;
    this.#applyState(this.#hovered ? 'hover' : 'idle');
    this.#onClick();
  }

  #handleGlobalUp() {
    if (this.#pressed && !this.#hovered) {
      this.#pressed = false;
      if (this.#enabled) {
        this.#applyState('idle');
      }
    }
  }

  /** @param {'idle' | 'hover' | 'pressed' | 'disabled'} state */
  #applyState(state) {
    const styles = {
      idle: { fill: COLORS.panel, stroke: COLORS.border, text: TEXT_COLORS.primary, scale: 1 },
      hover: { fill: COLORS.panelHover, stroke: COLORS.borderHover, text: TEXT_COLORS.accent, scale: HOVER_SCALE },
      pressed: { fill: COLORS.panelPressed, stroke: COLORS.borderHover, text: TEXT_COLORS.accent, scale: PRESSED_SCALE },
      disabled: { fill: COLORS.disabled, stroke: COLORS.disabled, text: TEXT_COLORS.disabled, scale: 1 },
    };
    const style = styles[state];
    this.#background.setFillStyle(style.fill).setStrokeStyle(2, style.stroke);
    this.#label.setColor(style.text);

    this.#scaleTween?.stop();
    this.#scaleTween = this.scene.tweens.add({
      targets: this,
      scale: style.scale,
      duration: SCALE_TWEEN_MS,
      ease: 'Quad.easeOut',
    });
  }

  #cleanup() {
    this.#scaleTween?.stop();
    this.#scaleTween = null;
    this.scene?.input?.off(Phaser.Input.Events.POINTER_UP, this.#handleGlobalUp, this);
  }
}
