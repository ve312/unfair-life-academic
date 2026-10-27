import Phaser from 'phaser';
import { bffClient } from '../api/bffClient.js';
import { playthroughHistory } from '../managers/history.js';
import { ChoiceBtn } from '../objects/ChoiceBtn.js';
import { StoryCard } from '../objects/StoryCard.js';
import { COLORS, FONTS, GAME_HEIGHT, GAME_WIDTH, TEXT_COLORS } from '../theme.js';

const COLUMNS = 3;
const ROWS = 2;
const CARDS_PER_PAGE = COLUMNS * ROWS;
const CARD_WIDTH = 370;
const CARD_HEIGHT = 230;
const GAP_X = 30;
const GAP_Y = 26;
const GRID_TOP = 140;
const PAGER_Y = 672;
const FADE_IN_MS = 400;
const ERROR_COLOR = '#e07a6a';
/** How long a failed-start message replaces the pager. */
const FOOTER_ERROR_MS = 4000;

/**
 * Story hub: lists the playable stories from GET /game/stories as a
 * paginated grid of StoryCards (3 x 2 per page), loads each visible card's
 * cover on the fly, and starts the chosen story through POST /game/start.
 *
 * Navigation: click a card to play, Prev / Next buttons or the arrow keys to
 * change page, Back or Escape to return to the title screen.
 */
export class StorySelection extends Phaser.Scene {
  static KEY = 'StorySelection';

  /** @type {import('../api/bffClient.js').StorySummary[]} */
  stories = [];
  /** @type {StoryCard[]} */
  cards = [];
  page = 0;
  /** @type {Phaser.GameObjects.Text | null} */
  #status = null;
  /** @type {Phaser.GameObjects.Text | null} */
  #footer = null;
  /** @type {Phaser.Time.TimerEvent | null} */
  #footerTimer = null;
  /** @type {ChoiceBtn | null} */
  #retryButton = null;
  /** @type {ChoiceBtn | null} */
  #backButton = null;
  /** @type {{ prev: ChoiceBtn, next: ChoiceBtn, label: Phaser.GameObjects.Text } | null} */
  #pager = null;
  /** @type {AbortController | null} */
  #request = null;
  #starting = false;
  /** Cover URLs that failed to load this session, so paging back does not retry them forever. */
  #failedCovers = new Set();
  /** Per-cover loader listeners, removed on shutdown so a restarted scene never stacks them. */
  #coverListeners = [];

  constructor() {
    super(StorySelection.KEY);
  }

  init() {
    this.stories = [];
    this.cards = [];
    this.page = 0;
    this.#request = null;
    this.#starting = false;
    this.#coverListeners = [];
    this.#footerTimer = null;
  }

  create() {
    const centerX = GAME_WIDTH / 2;
    this.cameras.main.setBackgroundColor(COLORS.background);
    this.cameras.main.fadeIn(FADE_IN_MS, 0, 0, 0);

    this.add
      .text(centerX, 58, 'Choose a Life', { fontFamily: FONTS.title, fontSize: '46px', color: TEXT_COLORS.primary })
      .setOrigin(0.5);
    this.add
      .text(centerX, 100, 'Each story starts somewhere different. Not every start is fair.', {
        fontFamily: FONTS.body,
        fontSize: '18px',
        color: TEXT_COLORS.muted,
      })
      .setOrigin(0.5);

    this.#backButton = new ChoiceBtn(this, 100, 58, '‹ Back', () => this.#goBack(), { width: 130, height: 48, fontSize: 18 });

    this.#status = this.add
      .text(centerX, GAME_HEIGHT / 2, '', {
        fontFamily: FONTS.body,
        fontSize: '22px',
        color: TEXT_COLORS.muted,
        align: 'center',
        wordWrap: { width: GAME_WIDTH - 300 },
      })
      .setOrigin(0.5)
      .setDepth(10);

    this.#pager = {
      prev: new ChoiceBtn(this, centerX - 190, PAGER_Y, '‹ Prev', () => this.#showPage(this.page - 1), { width: 150, height: 48, fontSize: 18 }),
      next: new ChoiceBtn(this, centerX + 190, PAGER_Y, 'Next ›', () => this.#showPage(this.page + 1), { width: 150, height: 48, fontSize: 18 }),
      label: this.add
        .text(centerX, PAGER_Y, '', { fontFamily: FONTS.body, fontSize: '18px', color: TEXT_COLORS.muted })
        .setOrigin(0.5),
    };
    this.#footer = this.add
      .text(centerX, PAGER_Y, '', { fontFamily: FONTS.body, fontSize: '20px', color: TEXT_COLORS.muted, align: 'center' })
      .setOrigin(0.5)
      .setDepth(10)
      .setVisible(false);
    this.#setPagerVisible(false);

    this.input.keyboard?.on('keydown-LEFT', this.#onKeyLeft, this);
    this.input.keyboard?.on('keydown-RIGHT', this.#onKeyRight, this);
    this.input.keyboard?.on('keydown-ESC', this.#goBack, this);
    this.load.on(Phaser.Loader.Events.FILE_LOAD_ERROR, this.#onCoverError, this);
    this.events.once(Phaser.Scenes.Events.SHUTDOWN, this.#onShutdown, this);

    this.#fetchStories();
  }

  // ------------------------------------------------------------- loading

  async #fetchStories() {
    this.#clearRetry();
    this.#setStatus('Despertando servidores...\nesto puede tomar hasta un minuto.');
    this.#request = new AbortController();
    let stories;
    // Only the request is guarded: a rendering bug must surface as a bug, not as a "server error".
    try {
      stories = await bffClient.getStories({ signal: this.#request.signal });
    } catch (error) {
      this.#request = null;
      if (error?.kind === 'aborted' || !this.sys.isActive()) {
        return;
      }
      console.error('[StorySelection] Could not load stories:', error);
      this.#setStatus(error?.message ?? 'Could not load the stories.', true);
      this.#showRetry('Try Again');
      return;
    }
    this.#request = null;
    if (!this.sys.isActive()) {
      return;
    }
    this.stories = stories;
    if (stories.length === 0) {
      this.#setStatus('No stories are available yet.\nAdd a story with a starting scene in the CMS.');
      this.#showRetry('Refresh');
      return;
    }
    this.#setStatus('');
    this.#showPage(0);
  }

  #loadCovers() {
    let queued = 0;
    for (const card of this.cards) {
      const url = card.story.coverImageUrl;
      if (!url || this.#failedCovers.has(url)) {
        continue;
      }
      const key = StorySelection.#coverKey(url);
      if (this.textures.exists(key)) {
        card.setCover(key);
        continue;
      }
      // One listener per card: a cover shared by two stories fills both cards.
      const event = `${Phaser.Loader.Events.FILE_KEY_COMPLETE}image-${key}`;
      const listener = () => {
        if (card.scene && this.cards.includes(card)) {
          card.setCover(key);
        }
      };
      this.load.once(event, listener);
      this.#coverListeners.push([event, listener]);
      // The loader ignores a key that is already queued, so a shared cover is fetched once.
      this.load.image(key, url);
      queued++;
    }
    if (queued > 0 && !this.load.isLoading()) {
      this.load.start();
    }
  }

  #onCoverError(file) {
    if (typeof file?.key === 'string' && file.key.startsWith('cover:')) {
      this.#failedCovers.add(file.key.slice('cover:'.length));
      console.warn(`[StorySelection] Cover failed to load, keeping the placeholder: ${file.url}`);
    }
  }

  // ------------------------------------------------------------- rendering

  #showPage(page) {
    const pageCount = Math.ceil(this.stories.length / CARDS_PER_PAGE);
    if (this.#starting || page < 0 || page >= pageCount) {
      return;
    }
    this.page = page;
    this.#footer.setVisible(false);
    this.#footerTimer?.remove(false);
    this.#footerTimer = null;
    this.cards.forEach((card) => card.destroy());

    const gridWidth = COLUMNS * CARD_WIDTH + (COLUMNS - 1) * GAP_X;
    const firstX = (GAME_WIDTH - gridWidth) / 2 + CARD_WIDTH / 2;
    const firstY = GRID_TOP + CARD_HEIGHT / 2;

    this.cards = this.stories.slice(page * CARDS_PER_PAGE, (page + 1) * CARDS_PER_PAGE).map((story, index) => {
      const column = index % COLUMNS;
      const row = Math.floor(index / COLUMNS);
      const card = new StoryCard(
        this,
        firstX + column * (CARD_WIDTH + GAP_X),
        firstY + row * (CARD_HEIGHT + GAP_Y),
        story,
        (selected) => this.#startStory(selected, card),
        { width: CARD_WIDTH, height: CARD_HEIGHT },
      );
      card.setAlpha(0);
      this.tweens.add({ targets: card, alpha: 1, duration: 220, delay: index * 40 });
      return card;
    });

    this.#setPagerVisible(pageCount > 1);
    this.#pager.label.setText(`Page ${page + 1} / ${pageCount}`);
    this.#pager.prev.setEnabled(page > 0);
    this.#pager.next.setEnabled(page < pageCount - 1);
    this.#loadCovers();
  }

  #setPagerVisible(visible) {
    this.#pager.prev.setVisible(visible);
    this.#pager.next.setVisible(visible);
    this.#pager.label.setVisible(visible);
  }

  #setStatus(message, isError = false) {
    this.#status.setText(message).setColor(isError ? ERROR_COLOR : TEXT_COLORS.muted);
  }

  #showRetry(label) {
    this.#clearRetry();
    this.#retryButton = new ChoiceBtn(this, GAME_WIDTH / 2, GAME_HEIGHT / 2 + 90, label, () => this.#fetchStories(), {
      width: 240,
      height: 60,
    });
  }

  #clearRetry() {
    this.#retryButton?.destroy();
    this.#retryButton = null;
  }

  // ------------------------------------------------------------- starting

  /**
   * @param {import('../api/bffClient.js').StorySummary} story
   * @param {StoryCard} selectedCard
   */
  async #startStory(story, selectedCard) {
    if (this.#starting) {
      return;
    }
    this.#hideFooter();
    this.#starting = true;
    this.#lockMenu(selectedCard);
    this.#showFooter(`Starting "${story.title}"...`);
    this.#request = new AbortController();

    try {
      const node = await bffClient.startStory(story.id, { signal: this.#request.signal });
      this.#request = null;
      if (this.sys.isActive()) {
        playthroughHistory.begin(story);
        this.scene.start('Decision', { node });
      }
    } catch (error) {
      this.#request = null;
      if (error?.kind === 'aborted' || !this.sys.isActive()) {
        return;
      }
      console.error(`[StorySelection] Could not start story ${story.id}:`, error);
      this.#starting = false;
      this.#unlockMenu();
      // The error takes the pager's row; the pager returns after a few seconds or on the next action.
      this.#showFooter(error?.message ?? 'Could not start this story.', true);
      this.#footerTimer = this.time.delayedCall(FOOTER_ERROR_MS, () => this.#hideFooter());
    }
  }

  /** A one-line message in the pager's row, with the pager hidden so they never overlap. */
  #showFooter(message, isError = false) {
    this.#footerTimer?.remove(false);
    this.#footerTimer = null;
    this.#setPagerVisible(false);
    this.#footer.setText(message).setColor(isError ? ERROR_COLOR : TEXT_COLORS.muted).setVisible(true);
  }

  #hideFooter() {
    this.#footerTimer?.remove(false);
    this.#footerTimer = null;
    if (!this.#footer.visible) {
      return;
    }
    this.#footer.setVisible(false);
    this.#setPagerVisible(Math.ceil(this.stories.length / CARDS_PER_PAGE) > 1);
  }

  #lockMenu(selectedCard) {
    this.cards.forEach((card) => {
      card.setEnabled(false);
      card.setHighlighted(card === selectedCard);
    });
    this.#setPagerVisible(false);
    this.#backButton.setEnabled(false);
  }

  #unlockMenu() {
    this.cards.forEach((card) => {
      card.setHighlighted(false);
      card.setEnabled(true);
    });
    this.#backButton.setEnabled(true);
  }

  // ------------------------------------------------------------- navigation

  #onKeyLeft() {
    this.#showPage(this.page - 1);
  }

  #onKeyRight() {
    this.#showPage(this.page + 1);
  }

  #goBack() {
    if (!this.#starting) {
      this.scene.start('MainMenu');
    }
  }

  #onShutdown() {
    this.#request?.abort();
    this.#request = null;
    this.input.keyboard?.off('keydown-LEFT', this.#onKeyLeft, this);
    this.input.keyboard?.off('keydown-RIGHT', this.#onKeyRight, this);
    this.input.keyboard?.off('keydown-ESC', this.#goBack, this);
    this.load.off(Phaser.Loader.Events.FILE_LOAD_ERROR, this.#onCoverError, this);
    this.#coverListeners.forEach(([event, listener]) => this.load.off(event, listener));
    this.#coverListeners = [];
    this.cards = [];
  }

  static #coverKey(url) {
    return `cover:${url}`;
  }
}
