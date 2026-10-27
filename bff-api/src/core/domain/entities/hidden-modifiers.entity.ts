import { InvalidStoryContentError } from '../errors/domain.error.js';

/**
 * Hidden modifiers: the narrative context that a decision injects into the
 * character before the AI evaluates it.
 *
 * Theoretical frame. The mobility model estimates p(x) = P(Y = 1 | X = x)
 * for a static census profile x. A story, however, is a sequence of choices,
 * and a choice changes the circumstances in which the next outcome is
 * decided: accepting a loan raises capital, emigrating can change the class
 * of work, years of study raise the schooling level. A hidden modifier
 * encodes that change as a transformation T_d of the profile, attached to
 * decision d in the CMS. The model then evaluates the counterfactual profile
 * x' = T_d(x) instead of x. The weight of a narrative decision therefore
 * becomes measurable, as the shift in log-odds it produces,
 * F(x') - F(x).
 *
 * A transformation combines two kinds of mutation, identified by the type of
 * the value:
 *
 *   numeric trait     + integer -> additive delta,      x'_j = x_j + delta_j
 *   categorical trait + string  -> absolute replacement, x'_j = c_j
 *
 * Example authored in Strapi: { "capital-gain": 5000, "age": 2, "workclass": "Private" }.
 *
 * A zero delta is the identity element of addition, so it is dropped while
 * parsing. Every new decision starts from the CMS template
 * { "capital-gain": 0, "capital-loss": 0, "age": 0 }, which therefore parses
 * to HiddenModifiers.NONE until an editor changes a number. This is what
 * allows the template on decisions that have no mobility check.
 *
 * The modifiers are "hidden" because they are never sent to the game client:
 * the player sees the consequence of a choice, not its parameters.
 *
 * This value object validates untrusted CMS content once, when a scene is
 * loaded, so that a malformed modifier surfaces as InvalidStoryContentError
 * at load time and never during play. It is immutable and carries no
 * behaviour beyond parsing. Applying it to a character is the job of the
 * context-injection domain service.
 */

/** Integer-valued traits that accept an additive delta. */
export const NUMERIC_TRAITS = ['age', 'educationNum', 'hoursPerWeek', 'capitalGain', 'capitalLoss'] as const;

/** Categorical traits that accept an absolute replacement (census spellings, e.g. 'Private'). */
export const CATEGORICAL_TRAITS = [
  'workclass',
  'education',
  'maritalStatus',
  'occupation',
  'relationship',
  'race',
  'sex',
  'nativeCountry',
] as const;

export type NumericTrait = (typeof NUMERIC_TRAITS)[number];
export type CategoricalTrait = (typeof CATEGORICAL_TRAITS)[number];
export type ModifiableTrait = NumericTrait | CategoricalTrait;

/** Mirrors the Character entity's limit on categorical values. */
const MAX_CATEGORY_LENGTH = 64;

/**
 * Canonical trait names indexed by a spelling-insensitive key. Authors may
 * write 'capital-gain', 'capital_gain' or 'capitalGain': all of them reduce to
 * 'capitalgain'. fnlwgt is absent on purpose, because it is a census sampling
 * weight and not a circumstance that a story can change.
 */
const TRAIT_BY_KEY: ReadonlyMap<string, ModifiableTrait> = new Map(
  [...NUMERIC_TRAITS, ...CATEGORICAL_TRAITS].map((trait) => [normalizeKey(trait), trait]),
);

function normalizeKey(key: string): string {
  return key.replace(/[-_\s]/g, '').toLowerCase();
}

/**
 * Resolves an authored key to the canonical trait it names, in any spelling
 * ('capital-gain', 'capital_gain', 'capitalGain'); undefined when it names none.
 * Shared with the story base-profile parser, so both CMS fields accept the same keys.
 */
export function traitForKey(key: string): ModifiableTrait | undefined {
  return TRAIT_BY_KEY.get(normalizeKey(key));
}

export function isNumericTrait(trait: ModifiableTrait): trait is NumericTrait {
  return (NUMERIC_TRAITS as readonly string[]).includes(trait);
}

export class HiddenModifiers {
  /** The identity transformation, used when a decision has no modifiers (null or {} in the CMS). */
  static readonly NONE = new HiddenModifiers(new Map(), new Map());

  private constructor(
    private readonly deltaByTrait: ReadonlyMap<NumericTrait, number>,
    private readonly replacementByTrait: ReadonlyMap<CategoricalTrait, string>,
  ) {
    Object.freeze(this);
  }

  /**
   * Parses and validates the raw `hidden_modifiers` JSON of a decision.
   *
   * @param raw - the value returned by Strapi; null and undefined mean "no modifiers".
   * @param context - prefix for error messages, naming the scene and the decision.
   * @returns the validated modifiers, or HiddenModifiers.NONE when there are none.
   * @throws InvalidStoryContentError when the value is not a JSON object, a key is not a
   *   modifiable trait (or two keys spell the same trait), or a value has the wrong type.
   */
  static parse(raw: unknown, context: string): HiddenModifiers {
    if (raw === null || raw === undefined) {
      return HiddenModifiers.NONE;
    }
    if (typeof raw !== 'object' || Array.isArray(raw)) {
      throw new InvalidStoryContentError(`${context}: expected a JSON object such as { "capital-gain": 5000 }`);
    }
    const deltas = new Map<NumericTrait, number>();
    const replacements = new Map<CategoricalTrait, string>();
    const seen = new Set<ModifiableTrait>();

    for (const [key, value] of Object.entries(raw as Record<string, unknown>)) {
      const trait = TRAIT_BY_KEY.get(normalizeKey(key));
      if (trait === undefined) {
        throw new InvalidStoryContentError(
          `${context}: '${key}' is not a modifiable trait (allowed: ${[...NUMERIC_TRAITS, ...CATEGORICAL_TRAITS].join(', ')})`,
        );
      }
      if (seen.has(trait)) {
        throw new InvalidStoryContentError(`${context}: '${key}' modifies '${trait}' a second time`);
      }
      seen.add(trait);

      if (isNumericTrait(trait)) {
        if (typeof value !== 'number' || !Number.isSafeInteger(value)) {
          throw new InvalidStoryContentError(`${context}: '${key}' is numeric, so its value must be an integer delta`);
        }
        // A zero delta is the identity, x_j + 0 = x_j: the CMS template {"capital-gain": 0, ...} changes nothing.
        if (value !== 0) {
          deltas.set(trait, value);
        }
      } else {
        if (typeof value !== 'string' || value.trim().length === 0 || value.trim().length > MAX_CATEGORY_LENGTH) {
          throw new InvalidStoryContentError(
            `${context}: '${key}' is categorical, so its value must be a non-empty string of at most ${MAX_CATEGORY_LENGTH} characters`,
          );
        }
        replacements.set(trait, value.trim());
      }
    }
    return deltas.size === 0 && replacements.size === 0 ? HiddenModifiers.NONE : new HiddenModifiers(deltas, replacements);
  }

  /** True for the identity transformation. */
  get isEmpty(): boolean {
    return this.deltaByTrait.size === 0 && this.replacementByTrait.size === 0;
  }

  /** Additive deltas, in authoring order. */
  get deltas(): ReadonlyArray<readonly [NumericTrait, number]> {
    return [...this.deltaByTrait];
  }

  /** Absolute replacements, in authoring order. */
  get replacements(): ReadonlyArray<readonly [CategoricalTrait, string]> {
    return [...this.replacementByTrait];
  }
}
