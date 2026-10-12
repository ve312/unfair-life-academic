import type { StoryNode } from '../../../core/domain/entities/story-node.entity.js';
import { isPublishedYear, mobilityIncomeThreshold } from '../../../core/domain/services/consumer-price-index.js';
import { DEFAULT_STORY_YEAR } from '../../../core/use-cases/resolve-decision.service.js';

export interface DecisionView {
  id: string;
  text: string;
  /** Lets the client show that this choice depends on the character's luck. */
  requiresMobilityCheck: boolean;
}

export interface AudioView {
  url: string;
  loop: boolean;
  defaultVolume: number;
  /** Seconds into the file where playback starts; null plays from 0. */
  startTime: number | null;
  /** Seconds into the file where playback stops (or loops back); null plays to the end. */
  endTime: number | null;
}

/** JSON shape of a scene sent to the Phaser client. */
export interface StoryNodeView {
  id: string;
  title: string;
  /** Narrative context of the scene, shown with the title; null when the editor wrote none. */
  context: string | null;
  backgroundMediaUrl: string | null;
  /** Fragment of a video background, in seconds; both null for the whole file (and for images). */
  mediaStartTime: number | null;
  mediaEndTime: number | null;
  audio: AudioView | null;
  timerDuration: number;
  isCinematic: boolean;
  isEnding: boolean;
  /** What "mobility" means in this scene's year, for narrative text. */
  economy: EconomyView;
  decisions: DecisionView[];
}

export interface EconomyView {
  /** Year the scene takes place in (DEFAULT_STORY_YEAR when the content has none). */
  year: number;
  /** The model's "$50K in 1994" in this year's US dollars. */
  mobilityIncomeThreshold: number;
  /** False when the year's CPI is projected (2.5%/yr) rather than published. */
  cpiPublished: boolean;
}

/**
 * Converts a StoryNode into the client-facing view.
 *
 * Decision targets (nextSceneId / failureSceneId) are deliberately left out:
 * the BFF resolves where a choice leads, so players cannot read ahead in the
 * story graph or see where a failed mobility check would send them.
 */
export function presentStoryNode(node: StoryNode): StoryNodeView {
  const year = node.year ?? DEFAULT_STORY_YEAR;
  return {
    id: node.id,
    title: node.title,
    context: node.context,
    backgroundMediaUrl: node.backgroundMediaUrl,
    mediaStartTime: node.backgroundTrim.startTime,
    mediaEndTime: node.backgroundTrim.endTime,
    audio:
      node.audio === null
        ? null
        : {
            url: node.audio.url,
            loop: node.audio.loop,
            defaultVolume: node.audio.defaultVolume,
            startTime: node.audioTrim.startTime,
            endTime: node.audioTrim.endTime,
          },
    timerDuration: node.timerDuration,
    isCinematic: node.isCinematic,
    isEnding: node.isEnding,
    economy: { year, mobilityIncomeThreshold: mobilityIncomeThreshold(year), cpiPublished: isPublishedYear(year) },
    decisions: node.decisions.map((decision) => ({
      id: decision.id,
      text: decision.text,
      requiresMobilityCheck: decision.requiresMobilityCheck,
    })),
  };
}
