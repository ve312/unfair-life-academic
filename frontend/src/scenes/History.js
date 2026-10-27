import Phaser from 'phaser';
import { playthroughHistory } from '../managers/history.js';
import { ChoiceBtn } from '../objects/ChoiceBtn.js';
import { coverFit } from '../objects/imageFit.js';
import { COLORS, FONTS, GAME_HEIGHT, GAME_WIDTH, TEXT_COLORS } from '../theme.js';

const PANEL_WIDTH = 1060;
const PANEL_HEIGHT = 620;
const ENTRIES_PER_PAGE = 3;
const ROW_WIDTH = 1000;
const ROW_HEIGHT = 138;
const ROW_GAP = 12;
const THUMB_WIDTH = 220;
const THUMB_HEIGHT = 124;
const VIDEO_EXTENSION = /\.(mp4|webm|ogv|mov)(\?.*)?$/i;

/**
 * Read-only "story so far" overlay, launched on top of the paused Decision
 * scene. It lists the scenes already passed in this playthrough with their
 * background, title and the choice that was made, three per page. Nothing
 * here can change the story: it only reads the frontend history.
 *
 * Close with the Close button, Escape or H; page with the buttons or arrows.
 */
export class History extends Phaser.Scene {
  static KEY = 'History';

  /** @type {import('../managers/history.js').HistoryEntry[]} */
  entries = [];
  page = 0;
  #returnTo = 'Decision';
  /** @type {Phaser.GameObjects.GameObject[]} */
  #pageObjects = [];
  /** @type {{ prev: ChoiceBtn, next: ChoiceBtn, label: Phaser.GameObjects.Text } | null} */
  #pager = null;

  constructor() {
    super(History.KEY);
  }

  /** @param {{ returnTo?: string }} data */
  init(data) {
    this.#returnTo = data?.returnTo ?? 'Decision';
    this.entries = playthroughHistory.passed;
    this.page = Math.max(0, Math.ceil(this.entries.length / ENTRIES_PER_PAGE) - 1);
    this.#pageObjects = [];
  }

  create() {
    const centerX = GAME_WIDTH / 2;
    const centerY = GAME_HEIGHT / 2;
    const panelTop = centerY - PANEL_HEIGHT / 2;

    // Swallows clicks so nothing reaches the paused scene underneath.
    this.add.rectangle(0, 0, GAME_WIDTH, GAME_HEIGHT, 0x000000, 0.82).setOrigin(0).setInteractive();
    this.add.rectangle(centerX, centerY, PANEL_WIDTH, PANEL_HEIGHT, COLORS.background).setStrokeStyle(2, COLORS.border);

    const storyTitle = playthroughHistory.story?.title;
    this.add
      .text(centerX, panelTop + 40, 'Your Story So Far', { fontFamily: FONTS.title, fontSize: '34px', color: TEXT_COLORS.primary })
      .setOrigin(0.5);
    this.add
      .text(centerX, panelTop + 76, storyTitle ? `${storyTitle} · past choices cannot be changed` : 'Past choices cannot be changed', {
        fontFamily: FONTS.body,
        fontSize: '16px',
        color: TEXT_COLORS.muted,
      })
      .setOrigin(0.5);

    new ChoiceBtn(this, centerX + PANEL_WIDTH / 2 - 80, panelTop + 40, 'Close', () => this.#close(), {
      width: 120,
      height: 44,
      fontSize: 18,
    });

    const pagerY = panelTop + PANEL_HEIGHT - 36;
    this.#pager = {
      prev: new ChoiceBtn(this, centerX - 170, pagerY, '‹ Earlier', () => this.#showPage(this.page - 1), { width: 150, height: 42, fontSize: 17 }),
      next: new ChoiceBtn(this, centerX + 170, pagerY, 'Later ›', () => this.#showPage(this.page + 1), { width: 150, height: 42, fontSize: 17 }),
      label: this.add.text(centerX, pagerY, '', { fontFamily: FONTS.body, fontSize: '16px', color: TEXT_COLORS.muted }).setOrigin(0.5),
    };

    this.input.keyboard?.on('keydown-ESC', this.#close, this);
    this.input.keyboard?.on('keydown-H', this.#close, this);
    this.input.keyboard?.on('keydown-LEFT', this.#onKeyLeft, this);
    this.input.keyboard?.on('keydown-RIGHT', this.#onKeyRight, this);
    this.events.once(Phaser.Scenes.Events.SHUTDOWN, this.#onShutdown, this);

    this.#showPage(this.page);
  }

  #showPage(page) {
    const pageCount = Math.max(1, Math.ceil(this.entries.length / ENTRIES_PER_PAGE));
    if (page < 0 || page >= pageCount) {
      return;
    }
    this.page = page;
    this.#pageObjects.forEach((object) => object.destroy());
    this.#pageObjects = [];

    const centerX = GAME_WIDTH / 2;
    const firstY = GAME_HEIGHT / 2 - PANEL_HEIGHT / 2 + 110 + ROW_HEIGHT / 2;

    if (this.entries.length === 0) {
      this.#pageObjects.push(
        this.add
          .text(centerX, GAME_HEIGHT / 2, 'Nothing has happened yet.', { fontFamily: FONTS.body, fontSize: '20px', color: TEXT_COLORS.muted })
          .setOrigin(0.5),
      );
    }

    this.entries.slice(page * ENTRIES_PER_PAGE, (page + 1) * ENTRIES_PER_PAGE).forEach((entry, index) => {
      this.#pageObjects.push(...this.#renderEntry(entry, centerX, firstY + index * (ROW_HEIGHT + ROW_GAP)));
    });

    const visible = pageCount > 1;
    this.#pager.prev.setVisible(visible).setEnabled(page > 0);
    this.#pager.next.setVisible(visible).setEnabled(page < pageCount - 1);
    this.#pager.label.setVisible(visible).setText(`${page + 1} / ${pageCount}`);
  }

  /** @param {import('../managers/history.js').HistoryEntry} entry */
  #renderEntry(entry, centerX, centerY) {
    const left = centerX - ROW_WIDTH / 2;
    const thumbX = left + 12 + THUMB_WIDTH / 2;
    const textX = left + 12 + THUMB_WIDTH + 24;
    const textWidth = ROW_WIDTH - THUMB_WIDTH - 60;
    const objects = [];

    objects.push(this.add.rectangle(centerX, centerY, ROW_WIDTH, ROW_HEIGHT, COLORS.panel).setStrokeStyle(1, COLORS.border));
    objects.push(this.#thumbnail(entry, thumbX, centerY));

    const meta = [`Scene ${entry.step}`];
    if (entry.year !== null) {
      meta.push(String(entry.year));
    }
    if (entry.isCinematic) {
      meta.push('cinematic');
    }
    objects.push(
      this.add
        .text(textX, centerY - ROW_HEIGHT / 2 + 16, meta.join(' · '), { fontFamily: FONTS.body, fontSize: '14px', color: TEXT_COLORS.muted })
        .setOrigin(0, 0),
    );
    objects.push(
      this.add
        .text(textX, centerY - ROW_HEIGHT / 2 + 38, entry.title, {
          fontFamily: FONTS.title,
          fontSize: '22px',
          color: TEXT_COLORS.primary,
          wordWrap: { width: textWidth, useAdvancedWrap: true },
          maxLines: 2,
        })
        .setOrigin(0, 0),
    );

    const { choice } = entry;
    let outcome;
    if (choice.automatic) {
      outcome = 'The story moved on.';
    } else {
      outcome = `You chose: ${choice.text}${choice.requiresMobilityCheck ? '   (fate was weighed)' : ''}`;
    }
    objects.push(
      this.add
        .text(textX, centerY + ROW_HEIGHT / 2 - 16, outcome, {
          fontFamily: FONTS.body,
          fontSize: '17px',
          color: choice.automatic ? TEXT_COLORS.muted : TEXT_COLORS.accent,
          wordWrap: { width: textWidth, useAdvancedWrap: true },
          maxLines: 1,
        })
        .setOrigin(0, 1),
    );
    return objects;
  }

  /** The scene's background if it is already in the texture cache, otherwise a labelled placeholder. */
  #thumbnail(entry, x, y) {
    const key = entry.backgroundMediaUrl ? `background:${entry.backgroundMediaUrl}` : null;
    if (key && this.textures.exists(key)) {
      return coverFit(this.add.image(x, y, key), THUMB_WIDTH, THUMB_HEIGHT);
    }
    const isVideo = entry.backgroundMediaUrl !== null && VIDEO_EXTENSION.test(entry.backgroundMediaUrl);
    const placeholder = this.add.container(x, y, [
      this.add.rectangle(0, 0, THUMB_WIDTH, THUMB_HEIGHT, COLORS.panelHover),
      this.add
        .text(0, 0, isVideo ? '▶ video' : 'no image', { fontFamily: FONTS.body, fontSize: '15px', color: TEXT_COLORS.muted })
        .setOrigin(0.5),
    ]);
    return placeholder;
  }

  #onKeyLeft() {
    this.#showPage(this.page - 1);
  }

  #onKeyRight() {
    this.#showPage(this.page + 1);
  }

  #close() {
    this.scene.resume(this.#returnTo);
    this.scene.stop();
  }

  #onShutdown() {
    this.input.keyboard?.off('keydown-ESC', this.#close, this);
    this.input.keyboard?.off('keydown-H', this.#close, this);
    this.input.keyboard?.off('keydown-LEFT', this.#onKeyLeft, this);
    this.input.keyboard?.off('keydown-RIGHT', this.#onKeyRight, this);
    this.#pageObjects = [];
  }
}
