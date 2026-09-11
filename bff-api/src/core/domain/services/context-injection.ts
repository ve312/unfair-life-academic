import { CHARACTER_LIMITS, Character, type CharacterProps } from '../entities/character.entity.js';
import type { HiddenModifiers, ModifiableTrait, NumericTrait } from '../entities/hidden-modifiers.entity.js';

/**
 * Dynamic context injection: applies a decision's hidden modifiers to the
 * player's character, producing the temporary profile that the AI evaluates.
 *
 * Architecture. The game client owns the base profile x and sends it with
 * every choice. The CMS attaches a transformation T_d (HiddenModifiers) to
 * each decision d. When a decision requires a mobility check, the BFF
 * evaluates p(T_d(x)) instead of p(x). The mutated profile is ephemeral: it
 * exists for one prediction, is never returned to the client, and never
 * replaces the base profile. Each decision is therefore assessed against the
 * same starting point, and decisions do not accumulate hidden state that
 * the player cannot see.
 *
 * Pipeline order inside ResolveDecisionService:
 *
 *   x (story-year dollars)  --T_d-->  x'  --deflate(year)-->  x'' (1994 dollars)  --POST /predict-->  p(x'')
 *
 * Modifiers are applied before deflation because authors write amounts in the
 * currency of the story's year ("you inherit $5,000" in a 2026 scene). The
 * deflator then converts the mutated amounts together with the rest of the
 * profile.
 *
 * Feasibility. A delta may push a trait outside the domain on which the model
 * was trained, for example +2 years for an 89-year-old (the maximum age is 90).
 * Rejecting the choice would end the player's game for an authoring edge
 * case, so the value is projected onto the feasible range instead,
 * x'_j = min(max(x_j + delta_j, lo_j), hi_j), and the projection is recorded
 * as `clamped` for the audit log. Replacements need no projection, because
 * any non-empty census spelling is valid (unknown categories simply carry no
 * statistical weight).
 */

/** One applied mutation, as recorded in the audit log. */
export interface TraitAdjustment {
  trait: ModifiableTrait;
  operation: 'add' | 'set';
  /** The authored delta (add) or replacement (set). */
  value: number | string;
  before: number | string;
  after: number | string;
  /** True when the sum fell outside the trait's feasible range and was projected onto it. */
  clamped: boolean;
}

export interface ContextInjection {
  /** The mutated, validated character; the base character itself when there are no modifiers. */
  character: Character;
  /** The mutations in application order (deltas, then replacements); empty when there are none. */
  adjustments: TraitAdjustment[];
}

/** Feasible range of each numeric trait: the model's training domain (capital amounts are only bounded below). */
const FEASIBLE_RANGE: Record<NumericTrait, { min: number; max: number }> = {
  age: CHARACTER_LIMITS.age,
  educationNum: CHARACTER_LIMITS.educationNum,
  hoursPerWeek: CHARACTER_LIMITS.hoursPerWeek,
  capitalGain: { min: 0, max: Number.MAX_SAFE_INTEGER },
  capitalLoss: { min: 0, max: Number.MAX_SAFE_INTEGER },
};

/**
 * Applies T_d to a character.
 *
 * The base character is never mutated (it is frozen). A new Character is
 * built from a copy of its properties, so the result passes through the
 * entity's own invariants, exactly like a profile received from the client.
 *
 * @param character - the base profile x, in story-year dollars.
 * @param modifiers - the decision's hidden modifiers T_d.
 * @returns the mutated profile x' and the list of applied adjustments.
 */
export function injectContext(character: Character, modifiers: HiddenModifiers): ContextInjection {
  if (modifiers.isEmpty) {
    return { character, adjustments: [] };
  }
  const props = copyTraits(character);
  const adjustments: TraitAdjustment[] = [];

  for (const [trait, delta] of modifiers.deltas) {
    const before = character[trait];
    const { min, max } = FEASIBLE_RANGE[trait];
    const sum = before + delta;
    const after = Math.min(Math.max(sum, min), max);
    props[trait] = after;
    adjustments.push({ trait, operation: 'add', value: delta, before, after, clamped: after !== sum });
  }
  for (const [trait, replacement] of modifiers.replacements) {
    const before = character[trait];
    props[trait] = replacement;
    adjustments.push({ trait, operation: 'set', value: replacement, before, after: replacement, clamped: false });
  }
  return { character: new Character(props), adjustments };
}

/** Copies the 14 traits explicitly, so that a trait added to Character later cannot be dropped silently. */
function copyTraits(character: Character): CharacterProps {
  return {
    age: character.age,
    workclass: character.workclass,
    education: character.education,
    educationNum: character.educationNum,
    maritalStatus: character.maritalStatus,
    occupation: character.occupation,
    relationship: character.relationship,
    race: character.race,
    sex: character.sex,
    hoursPerWeek: character.hoursPerWeek,
    nativeCountry: character.nativeCountry,
    capitalGain: character.capitalGain,
    capitalLoss: character.capitalLoss,
    fnlwgt: character.fnlwgt,
  };
}
