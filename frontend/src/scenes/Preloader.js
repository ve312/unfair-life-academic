import Phaser from 'phaser';
import { COLORS, FONTS, GAME_HEIGHT, GAME_WIDTH, TEXT_COLORS } from '../theme.js';

const BAR_WIDTH = 420;
const BAR_HEIGHT = 8;

/**
 * Boot scene: shows a loading screen while shared assets load, then opens the
 * main menu. Story media (backgrounds, music) is loaded per scene at runtime
 * from the CMS, so this stays light.
 */
export class Preloader extends Phaser.Scene {
  static KEY = 'Preloader';

  constructor() {
    super(Preloader.KEY);
  }

  preload() {
    const centerX = GAME_WIDTH / 2;
    const centerY = GAME_HEIGHT / 2;

    this.cameras.main.setBackgroundColor(COLORS.background);
    this.add
      .text(centerX, centerY - 40, 'Loading...', {
        fontFamily: FONTS.title,
        fontSize: '36px',
        color: TEXT_COLORS.primary,
      })
      .setOrigin(0.5);

    const track = this.add.rectangle(centerX, centerY + 20, BAR_WIDTH, BAR_HEIGHT, COLORS.timerTrack);
    const fill = this.add
      .rectangle(track.x - BAR_WIDTH / 2, track.y, 0, BAR_HEIGHT, COLORS.accent)
      .setOrigin(0, 0.5);

    this.load.on(Phaser.Loader.Events.PROGRESS, (value) => {
      fill.width = BAR_WIDTH * value;
    });
    this.load.on(Phaser.Loader.Events.FILE_LOAD_ERROR, (file) => {
      console.error(`[Preloader] Failed to load '${file.key}' from ${file.url}`);
    });
  }

  create() {
    this.scene.start('MainMenu');
  }
}
