import type { StorySummary } from '../../domain/entities/story-summary.entity.js';

/** Injection token for ListStoriesUseCase. */
export const LIST_STORIES_USE_CASE = Symbol('ListStoriesUseCase');

/** Inbound port: the stories the player can choose from in the menu. */
export interface ListStoriesUseCase {
  /** Resolves an empty list when no story is playable yet. */
  execute(): Promise<StorySummary[]>;
}
