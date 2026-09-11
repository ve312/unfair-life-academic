import { Inject, Injectable, Logger } from '@nestjs/common';
import type { StoryNode } from '../domain/entities/story-node.entity.js';
import { BaseProfileRequiredError, DecisionNotFoundError } from '../domain/errors/domain.error.js';
import { injectContext, type TraitAdjustment } from '../domain/services/context-injection.js';
import { deflateToModelBaseYear } from '../domain/services/inflation-adjuster.js';
import { resolveMobility } from '../domain/services/systemic-luck.js';
import type { ResolveDecisionUseCase } from '../ports/in/resolve-decision.use-case.js';
import { AI_PREDICTOR_PORT, type AiPredictorPort } from '../ports/out/ai-predictor.port.js';
import { RANDOM_SOURCE, type RandomSource } from '../ports/out/random-source.port.js';
import { STORY_CONTENT_PORT, type StoryContentPort } from '../ports/out/story-content.port.js';

/** The game's present: used when content has no year, so money is never read as 1994 dollars by accident. */
export const DEFAULT_STORY_YEAR = 2026;

/**
 * Resolves a player's choice into the next scene of the story graph.
 *
 * A plain decision is a deterministic edge. A decision that requires a
 * mobility check is a fork, and its branch is chosen in five stages:
 *
 * 0. Base profile. The starting character x0 is the base_profile that the
 *    node's story declares in the CMS. It is server-side state, and the
 *    client never supplies it. The designer's calibration therefore holds for
 *    every player, and no client can submit a more favourable profile. The
 *    profile is optional for a story, because purely narrative stories never
 *    consult the AI. A mobility check in a story without one fails with a
 *    clear BaseProfileRequiredError instead of evaluating an arbitrary
 *    character.
 * 1. Dynamic context injection. The decision's hidden modifiers T_d, authored
 *    in Strapi, are applied to a copy of the base profile, giving the
 *    temporary profile x' = T_d(x0). The base profile is never modified or
 *    returned.
 * 2. Temporal normalisation. The money-valued traits of x' are deflated from
 *    the story year's dollars to 1994 dollars, the currency of the training
 *    census, giving x''.
 * 3. Inference. x'' is sent to the AI service (POST /predict), which returns
 *    the calibrated probability p(x'') and its selective-classification zone.
 * 4. Systemic luck. Outside the uncertainty band the zone decides the
 *    outcome. Inside it, a Bernoulli(p) draw does.
 *
 * Why the context is injected. Without stage 1 the model would evaluate
 * every mobility check of a story against the same static profile, so two
 * different narrative choices made by the same character would carry
 * exactly the same probability. Injecting the decision's consequences lets
 * the model price each choice: the specific weight of decision d is the
 * shift in log-odds, F(T_d(x)) - F(x), that its modifiers produce. Because the
 * deployed model is calibrated, that shift translates directly into a change
 * in the chance of mobility that the player experiences.
 *
 * Auditability. Every mobility check emits one audit log line containing
 * the decision, the year, every applied modifier (with its before and after
 * values and whether it was clamped), the exact profile passed to the AI
 * port (the adapter only renames its fields to the snake_case of
 * POST /predict and omits a null fnlwgt), the returned probability and zone, the luck roll when there was one, and
 * the result. From this line an auditor can replay the decision offline with
 * the training pipeline's artifacts and confirm that the live game followed
 * the documented policy. This closes the traceability gap between the
 * simulated results in traceability_report.json and real play.
 */
@Injectable()
export class ResolveDecisionService implements ResolveDecisionUseCase {
  private readonly logger = new Logger(ResolveDecisionService.name);

  constructor(
    @Inject(STORY_CONTENT_PORT) private readonly storyContent: StoryContentPort,
    @Inject(AI_PREDICTOR_PORT) private readonly aiPredictor: AiPredictorPort,
    @Inject(RANDOM_SOURCE) private readonly random: RandomSource,
  ) {}

  /**
   * Resolves a decision taken in a story node.
   *
   * @param currentNodeId - the node the player is in.
   * @param decisionId - the decision the player chose; it must be offered by that node.
   * @returns the next story node.
   * @throws DecisionNotFoundError when the node does not offer the decision.
   * @throws BaseProfileRequiredError when a mobility check belongs to a story without a base_profile.
   */
  async execute(currentNodeId: string, decisionId: string): Promise<StoryNode> {
    const currentNode = await this.storyContent.getStoryNode(currentNodeId);

    const decision = currentNode.findDecision(decisionId);
    if (decision === undefined) {
      throw new DecisionNotFoundError(currentNode.id, decisionId);
    }

    // Only a mobility check consults the AI; every other decision is deterministic.
    let hasMobility = false;
    if (decision.requiresMobilityCheck) {
      // Stage 0: x0 is the story's base profile. Without one, the AI has no character to evaluate.
      const baseProfile = currentNode.baseProfile;
      if (baseProfile === null) {
        throw new BaseProfileRequiredError(decision.id, currentNode.id);
      }
      const year = currentNode.year ?? DEFAULT_STORY_YEAR;
      // Stage 1: x' = T_d(x0). An absent modifier set (null in the CMS) is the identity transformation.
      const injection = injectContext(baseProfile, decision.hiddenModifiers);
      // Stage 2: x'' = deflate(x', year). Modifiers are authored in story-year dollars, so they are deflated too.
      const modelInput = deflateToModelBaseYear(injection.character, year);
      // Stages 3 and 4.
      const assessment = await this.aiPredictor.assessMobility(modelInput);
      const outcome = resolveMobility(assessment, this.random);
      hasMobility = outcome.hasMobility;
      this.logger.log(
        `Mobility check node=${currentNode.id} decision=${decision.id} year=${year} ` +
          `p=${assessment.probability.toFixed(4)} zone=${assessment.zone} decidedBy=${outcome.decidedBy}` +
          `${outcome.roll === null ? '' : ` roll=${outcome.roll.toFixed(4)}`} result=${hasMobility ? 'mobility' : 'no_mobility'} ` +
          `modifiers=${describeAdjustments(injection.adjustments)} modelInput=${JSON.stringify(modelInput)}`,
      );
    }

    return this.storyContent.getStoryNode(decision.resolveNextSceneId(hasMobility));
  }
}

/**
 * Serialises the applied modifiers for the audit log: `none`, or a JSON array
 * with one entry per mutation, e.g.
 * [{"trait":"capitalGain","operation":"add","value":5000,"before":0,"after":5000,"clamped":false}].
 */
function describeAdjustments(adjustments: TraitAdjustment[]): string {
  return adjustments.length === 0 ? 'none' : JSON.stringify(adjustments);
}
