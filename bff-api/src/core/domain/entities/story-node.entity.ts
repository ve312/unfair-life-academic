import { InvalidStoryContentError } from '../errors/domain.error.js';
import { isSupportedStoryYear, STORY_YEAR_RANGE } from '../services/consumer-price-index.js';
import { AudioConfig } from './audio-config.entity.js';
import { Character } from './character.entity.js';
import { MediaTrim } from './media-trim.entity.js';
import { Decision } from './decision.entity.js';

/** Upper bound for a scene's decision timer, matching the CMS schema (seconds). */
export const MAX_TIMER_DURATION_SECONDS = 600;

export interface StoryNodeProps {
  id: string;
  title: string;
  /** Full narrative context written by the editor; null (or blank) when the scene has none. */
  context?: string | null;
  backgroundMediaUrl?: string | null;
  audio?: AudioConfig | null;
  /** Fragment of the background video to play; images ignore it. */
  backgroundTrim?: MediaTrim;
  /** Fragment of the audio track to play. */
  audioTrim?: MediaTrim;
  timerDuration: number;
  isCinematic: boolean;
  /** Year the scene takes place in (its own, or its story's); null when the content has none. */
  year?: number | null;
  /** Starting character of the story the scene belongs to; null when the story declares none. */
  baseProfile?: Character | null;
  decisions: Decision[];
}

/**
 * A scene of the story graph as the game client renders it: what is shown,
 * what is heard, how long the player has to choose, and the choices offered.
 * A node with no decisions is an ending.
 */
export class StoryNode {
  readonly id: string;
  readonly title: string;
  /** Narrative context shown with the title: who the character is now, what is happening, what is at stake. */
  readonly context: string | null;
  readonly backgroundMediaUrl: string | null;
  readonly audio: AudioConfig | null;
  readonly backgroundTrim: MediaTrim;
  readonly audioTrim: MediaTrim;
  /** Seconds the player has to choose; 0 means no timer. */
  readonly timerDuration: number;
  /** Cinematic scenes play automatically instead of showing the decision UI. */
  readonly isCinematic: boolean;
  /** Year the scene takes place in; drives the inflation adjustment of mobility checks. */
  readonly year: number | null;
  /**
   * The story's base profile x0, on which the mobility checks of this scene are evaluated.
   * Server-side state: the presenter never sends it to the client.
   */
  readonly baseProfile: Character | null;
  readonly decisions: readonly Decision[];

  constructor(props: StoryNodeProps) {
    this.id = StoryNode.requiredText('id', props.id);
    this.title = StoryNode.requiredText('title', props.title, this.id);
    this.context = StoryNode.optionalProse('context', props.context, this.id);
    this.backgroundMediaUrl = StoryNode.optionalText('backgroundMediaUrl', props.backgroundMediaUrl, this.id);

    if (props.audio != null && !(props.audio instanceof AudioConfig)) {
      throw new InvalidStoryContentError(`Story node '${this.id}': 'audio' must be an AudioConfig or null`);
    }
    this.audio = props.audio ?? null;

    for (const [name, trim] of [['backgroundTrim', props.backgroundTrim], ['audioTrim', props.audioTrim]] as const) {
      if (trim !== undefined && !(trim instanceof MediaTrim)) {
        throw new InvalidStoryContentError(`Story node '${this.id}': '${name}' must be a MediaTrim`);
      }
    }
    this.backgroundTrim = props.backgroundTrim ?? MediaTrim.NONE;
    this.audioTrim = props.audioTrim ?? MediaTrim.NONE;

    if (
      !Number.isInteger(props.timerDuration) ||
      props.timerDuration < 0 ||
      props.timerDuration > MAX_TIMER_DURATION_SECONDS
    ) {
      throw new InvalidStoryContentError(
        `Story node '${this.id}': 'timerDuration' must be an integer between 0 and ${MAX_TIMER_DURATION_SECONDS}`,
      );
    }
    this.timerDuration = props.timerDuration;

    if (typeof props.isCinematic !== 'boolean') {
      throw new InvalidStoryContentError(`Story node '${this.id}': 'isCinematic' must be a boolean`);
    }
    this.isCinematic = props.isCinematic;

    if (props.year != null && !isSupportedStoryYear(props.year)) {
      throw new InvalidStoryContentError(
        `Story node '${this.id}': 'year' must be an integer between ${STORY_YEAR_RANGE.min} and ${STORY_YEAR_RANGE.max}`,
      );
    }
    this.year = props.year ?? null;

    if (props.baseProfile != null && !(props.baseProfile instanceof Character)) {
      throw new InvalidStoryContentError(`Story node '${this.id}': 'baseProfile' must be a Character or null`);
    }
    this.baseProfile = props.baseProfile ?? null;

    if (!Array.isArray(props.decisions) || props.decisions.some((decision) => !(decision instanceof Decision))) {
      throw new InvalidStoryContentError(`Story node '${this.id}': 'decisions' must be an array of Decision entities`);
    }
    const ids = props.decisions.map((decision) => decision.id);
    const duplicate = ids.find((id, index) => ids.indexOf(id) !== index);
    if (duplicate !== undefined) {
      throw new InvalidStoryContentError(`Story node '${this.id}' contains decision '${duplicate}' more than once`);
    }
    this.decisions = Object.freeze([...props.decisions]);
    Object.freeze(this);
  }

  get isEnding(): boolean {
    return this.decisions.length === 0;
  }

  get hasTimer(): boolean {
    return this.timerDuration > 0;
  }

  findDecision(decisionId: string): Decision | undefined {
    return this.decisions.find((decision) => decision.id === decisionId);
  }

  private static requiredText(name: string, value: unknown, nodeId?: string): string {
    if (typeof value !== 'string' || value.trim().length === 0) {
      const owner = nodeId ? `Story node '${nodeId}': ` : 'Story node: ';
      throw new InvalidStoryContentError(`${owner}'${name}' must be a non-empty string`);
    }
    return value.trim();
  }

  private static optionalText(name: string, value: unknown, nodeId: string): string | null {
    if (value == null) {
      return null;
    }
    return StoryNode.requiredText(name, value, nodeId);
  }

  /**
   * Free text written by an editor. Strapi returns '' when a text field is cleared, so blank
   * text means "none" instead of being rejected; line breaks inside the text are kept.
   */
  private static optionalProse(name: string, value: unknown, nodeId: string): string | null {
    if (value == null) {
      return null;
    }
    if (typeof value !== 'string') {
      throw new InvalidStoryContentError(`Story node '${nodeId}': '${name}' must be text`);
    }
    const trimmed = value.trim();
    return trimmed.length === 0 ? null : trimmed;
  }
}
