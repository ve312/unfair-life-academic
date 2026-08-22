import { InvalidStoryContentError } from '../errors/domain.error.js';
import { HiddenModifiers } from './hidden-modifiers.entity.js';

export interface DecisionProps {
  id: string;
  text: string;
  requiresMobilityCheck: boolean;
  nextSceneId: string;
  failureSceneId?: string | null;
  /** Narrative context injected into the character before the mobility check; none when omitted. */
  hiddenModifiers?: HiddenModifiers | null;
}

/**
 * A choice offered in a scene: a directed edge of the story graph.
 *
 * A plain decision always leads to nextSceneId. A decision that requires a
 * mobility check is a fork: nextSceneId on success, failureSceneId on
 * failure, so a mobility check must always have a failure scene.
 *
 * A mobility check may also carry hidden modifiers, the transformation T_d
 * that the decision applies to the character's circumstances before the AI
 * evaluates them (see HiddenModifiers and injectContext). Modifiers only take
 * effect through a mobility check, so a plain decision that declares them is
 * rejected as an authoring error instead of being silently ignored.
 */
export class Decision {
  readonly id: string;
  readonly text: string;
  readonly requiresMobilityCheck: boolean;
  readonly nextSceneId: string;
  readonly failureSceneId: string | null;
  readonly hiddenModifiers: HiddenModifiers;

  constructor(props: DecisionProps) {
    this.id = Decision.requiredText('id', props.id);
    this.text = Decision.requiredText('text', props.text);
    if (typeof props.requiresMobilityCheck !== 'boolean') {
      throw new InvalidStoryContentError(`Decision '${this.id}': 'requiresMobilityCheck' must be a boolean`);
    }
    this.requiresMobilityCheck = props.requiresMobilityCheck;
    this.nextSceneId = Decision.requiredText('nextSceneId', props.nextSceneId, this.id);
    this.failureSceneId =
      props.failureSceneId == null ? null : Decision.requiredText('failureSceneId', props.failureSceneId, this.id);

    if (this.requiresMobilityCheck && this.failureSceneId === null) {
      throw new InvalidStoryContentError(
        `Decision '${this.id}' requires a mobility check but has no failure scene to fork to`,
      );
    }
    this.hiddenModifiers = props.hiddenModifiers ?? HiddenModifiers.NONE;
    if (!this.requiresMobilityCheck && !this.hiddenModifiers.isEmpty) {
      throw new InvalidStoryContentError(
        `Decision '${this.id}' declares hidden modifiers but no mobility check, so they would never take effect`,
      );
    }
    Object.freeze(this);
  }

  /**
   * Returns the scene the player moves to after taking this decision.
   * hasMobility is only consulted for decisions that require a mobility check.
   */
  resolveNextSceneId(hasMobility: boolean): string {
    if (!this.requiresMobilityCheck) {
      return this.nextSceneId;
    }
    return hasMobility ? this.nextSceneId : (this.failureSceneId as string);
  }

  private static requiredText(name: string, value: unknown, decisionId?: string): string {
    if (typeof value !== 'string' || value.trim().length === 0) {
      const owner = decisionId ? `Decision '${decisionId}': ` : 'Decision: ';
      throw new InvalidStoryContentError(`${owner}'${name}' must be a non-empty string`);
    }
    return value.trim();
  }
}
