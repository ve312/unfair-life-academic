import { InvalidCharacterError, InvalidStoryContentError } from '../errors/domain.error.js';
import { Character, type CharacterProps } from './character.entity.js';
import { CATEGORICAL_TRAITS, NUMERIC_TRAITS, traitForKey, type ModifiableTrait } from './hidden-modifiers.entity.js';

/**
 * Story base profile: the character every playthrough of a story starts as.
 *
 * Narrative coherence requires a single, authoritative starting point. Each
 * story declares in the CMS (field `base_profile`) the census profile x0 of
 * its protagonist, for example a 19-year-old farmhand born in Guatemala.
 * Every mobility check of that story evaluates the model on x0 transformed by
 * the chosen decision's hidden modifiers, p(T_d(x0)). The profile belongs to
 * the content, not to the client, for two reasons:
 *
 * - Coherence. The same story always starts from the same character, so the
 *   probabilities that the designer calibrated while writing the story are
 *   the ones that players experience.
 * - Integrity. A client cannot submit a more favourable profile to obtain a
 *   better outcome; the server decides who the character is.
 *
 * The JSON holds exactly the 13 traits of the model, in any key spelling
 * (the same rule as hidden_modifiers). fnlwgt is excluded: it is a census
 * sampling weight that the model does not use.
 */

const REQUIRED_TRAITS: readonly ModifiableTrait[] = [...NUMERIC_TRAITS, ...CATEGORICAL_TRAITS];

/**
 * Parses and validates the raw `base_profile` JSON of a story.
 *
 * @param raw - the value returned by Strapi; null, undefined and {} mean the story declares no profile
 *   (the field is optional: purely narrative stories never consult the AI).
 * @param context - prefix for error messages, naming the story.
 * @returns the validated Character, or null when the story declares none.
 * @throws InvalidStoryContentError when the value is not a JSON object, a key is unknown or
 *   repeated, a trait is missing, or a value breaks a Character invariant (type or range).
 */
export function parseBaseProfile(raw: unknown, context: string): Character | null {
  if (raw === null || raw === undefined) {
    return null;
  }
  if (typeof raw !== 'object' || Array.isArray(raw)) {
    throw new InvalidStoryContentError(`${context}: expected a JSON object with the 13 traits of the character`);
  }
  if (Object.keys(raw).length === 0) {
    return null; // An editor who empties the optional field leaves {}: the story declares no profile.
  }

  const values = new Map<ModifiableTrait, unknown>();
  for (const [key, value] of Object.entries(raw as Record<string, unknown>)) {
    const trait = traitForKey(key);
    if (trait === undefined) {
      throw new InvalidStoryContentError(`${context}: '${key}' is not a trait of the model (allowed: ${REQUIRED_TRAITS.join(', ')})`);
    }
    if (values.has(trait)) {
      throw new InvalidStoryContentError(`${context}: '${key}' defines '${trait}' a second time`);
    }
    values.set(trait, value);
  }
  const missing = REQUIRED_TRAITS.filter((trait) => !values.has(trait));
  if (missing.length > 0) {
    throw new InvalidStoryContentError(`${context}: missing traits ${missing.join(', ')}`);
  }

  try {
    return new Character(Object.fromEntries(values) as unknown as CharacterProps);
  } catch (error) {
    if (error instanceof InvalidCharacterError) {
      throw new InvalidStoryContentError(`${context}: ${error.message}`);
    }
    throw error;
  }
}
