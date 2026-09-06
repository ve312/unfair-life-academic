import type { StoryNode } from '../../domain/entities/story-node.entity.js';
import type { StorySummary } from '../../domain/entities/story-summary.entity.js';

/** Injection token for StoryContentPort (a runtime value, since interfaces vanish at compile time). */
export const STORY_CONTENT_PORT = Symbol('StoryContentPort');

/**
 * Outbound port: where the authored story graph comes from (e.g. Strapi).
 *
 * Implementations must reject with StoryNodeNotFoundError when a node does
 * not exist, and with InvalidStoryContentError when stored content cannot be
 * turned into valid domain entities.
 */
export interface StoryContentPort {
  /** Loads one scene of the story graph by its id. */
  getStoryNode(nodeId: string): Promise<StoryNode>;

  /**
   * Loads the scene a story begins with. Rejects with StoryNotFoundError when
   * the story does not exist, and with InvalidStoryContentError when it has
   * no starting scene.
   */
  getStartingNode(storyId: string): Promise<StoryNode>;

  /**
   * Lists the stories a player can start: only those with a starting scene,
   * ordered by title. Entries whose content is malformed are left out (and
   * reported by the implementation) so one broken story cannot hide the rest.
   */
  listStories(): Promise<StorySummary[]>;
}
