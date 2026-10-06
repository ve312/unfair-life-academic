import Phaser from 'phaser';
import { Decision } from './scenes/Decision.js';
import { History } from './scenes/History.js';
import { MainMenu } from './scenes/MainMenu.js';
import { Preloader } from './scenes/Preloader.js';
import { StorySelection } from './scenes/StorySelection.js';
import { COLORS, GAME_HEIGHT, GAME_WIDTH } from './theme.js';

/** @type {Phaser.Types.Core.GameConfig} */
const config = {
  type: Phaser.AUTO,
  parent: 'game',
  width: GAME_WIDTH,
  height: GAME_HEIGHT,
  backgroundColor: COLORS.background,
  scale: {
    mode: Phaser.Scale.FIT,
    autoCenter: Phaser.Scale.CENTER_BOTH,
  },
  // Howler.js owns all audio (see managers/audio.js), so Phaser's sound system is disabled.
  audio: { noAudio: true },
  // Story media is served by the CMS on another origin; WebGL textures need CORS-enabled loads.
  loader: { crossOrigin: 'anonymous' },
  scene: [Preloader, MainMenu, StorySelection, Decision, History],
};

// Render's free tier puts the AI service to sleep; waking it while the player is still in the menus
// keeps the first mobility check from timing out (a 502 "invalid response"). Silent: no-cors, errors ignored.
const AI_WAKE_URL = import.meta.env?.VITE_AI_WAKE_URL || 'https://unafirlifevc.onrender.com/';
fetch(AI_WAKE_URL, { method: 'HEAD', mode: 'no-cors' }).catch(() => {});

const game = new Phaser.Game(config);

if (import.meta.hot) {
  // Avoid stacking canvases when Vite hot-reloads this module during development.
  import.meta.hot.dispose(() => game.destroy(true));
}

export default game;
