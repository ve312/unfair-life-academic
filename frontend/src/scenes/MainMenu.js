import Phaser from 'phaser';
import { ChoiceBtn } from '../objects/ChoiceBtn.js';
import { COLORS, FONTS, GAME_HEIGHT, GAME_WIDTH, TEXT_COLORS } from '../theme.js';

const FADE_IN_MS = 600;

/** Title screen: the entry point that leads to the story-selection hub. */
export class MainMenu extends Phaser.Scene {
  static KEY = 'MainMenu';

  /** @type {ChoiceBtn | null} */
  startButton = null;

  constructor() {
    super(MainMenu.KEY);
  }

  create() {
    const centerX = GAME_WIDTH / 2;

    this.cameras.main.setBackgroundColor(COLORS.background);
    this.cameras.main.fadeIn(FADE_IN_MS, 0, 0, 0);

    this.add
      .text(centerX, GAME_HEIGHT * 0.32, 'Unfair Life', {
        fontFamily: FONTS.title,
        fontSize: '96px',
        color: TEXT_COLORS.primary,
      })
      .setOrigin(0.5)
      .setShadow(0, 6, '#000000', 12, false, true);

    this.add
      .text(centerX, GAME_HEIGHT * 0.32 + 82, 'Where you start shapes where you can go.', {
        fontFamily: FONTS.body,
        fontSize: '24px',
        color: TEXT_COLORS.muted,
      })
      .setOrigin(0.5);

    this.add.rectangle(centerX, GAME_HEIGHT * 0.32 + 130, 160, 2, COLORS.accent);

    this.startButton = new ChoiceBtn(this, centerX, GAME_HEIGHT * 0.68, 'Start Game', () => this.scene.start('StorySelection'), {
      width: 360,
      height: 76,
      fontSize: 28,
    });
  }
}
