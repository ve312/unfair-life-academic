import type { StoryNode } from '../../domain/entities/story-node.entity.js';

/** Injection token for ResolveDecisionUseCase. */
export const RESOLVE_DECISION_USE_CASE = Symbol('ResolveDecisionUseCase');

/** Inbound port: apply the player's choice and return the scene it leads to. */
export interface ResolveDecisionUseCase {
  /**
   * A mobility check runs on the base profile of the node's story, which is
   * server-side content: the client never supplies the character. Decisions
   * without a mobility check never need a profile.
   *
   * Rejects with StoryNodeNotFoundError for an unknown node,
   * DecisionNotFoundError when the decision is not offered in that node, and
   * BaseProfileRequiredError when a mobility check belongs to a story without
   * a base_profile.
   */
  execute(currentNodeId: string, decisionId: string): Promise<StoryNode>;
}
