import type { StoryNode } from '../../domain/entities/story-node.entity.js';

/** Injection token for StartStoryUseCase. */
export const START_STORY_USE_CASE = Symbol('StartStoryUseCase');

/** Inbound port: begin a story and return the first scene to play. */
export interface StartStoryUseCase {
  /**
   * Rejects with StoryNotFoundError when the story does not exist, and with
   * InvalidStoryContentError when its content is incomplete or malformed.
   */
  execute(storyId: string): Promise<StoryNode>;
}
