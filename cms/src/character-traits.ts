/**
 * The character traits of the mobility model, shared by the CMS validations
 * (decision hidden_modifiers and story base_profile).
 *
 * Mirrors the BFF (bff-api/src/core/domain/entities/character.entity.ts and
 * hidden-modifiers.entity.ts), so that anything the admin panel accepts can
 * always be played. Keys may be written in any spelling: 'capital-gain',
 * 'capital_gain' and 'capitalGain' all name the same trait.
 *
 * fnlwgt is deliberately absent. It is a census sampling weight, not a
 * circumstance of the character, and the model does not use it.
 */

/** Integer traits, with the ranges the model saw in training (capital amounts are only bounded below). */
export const NUMERIC_TRAITS = {
  age: { min: 17, max: 90 },
  educationNum: { min: 1, max: 16 },
  hoursPerWeek: { min: 1, max: 99 },
  capitalGain: { min: 0, max: Number.MAX_SAFE_INTEGER },
  capitalLoss: { min: 0, max: Number.MAX_SAFE_INTEGER },
} as const;

/** Text traits, using the census spellings (e.g. 'Private', 'HS-grad', 'Mexico'). */
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

/** Same limit as the BFF's Character entity. */
export const MAX_CATEGORY_LENGTH = 64;

export type NumericTrait = keyof typeof NUMERIC_TRAITS;
export type CategoricalTrait = (typeof CATEGORICAL_TRAITS)[number];
export type Trait = NumericTrait | CategoricalTrait;

export const ALL_TRAITS: readonly Trait[] = [...(Object.keys(NUMERIC_TRAITS) as NumericTrait[]), ...CATEGORICAL_TRAITS];

/** 'capital-gain', 'capital_gain' and 'capitalGain' all reduce to 'capitalgain'. */
export function normalizeKey(key: string): string {
  return key.replace(/[-_\s]/g, '').toLowerCase();
}

const TRAIT_BY_KEY: ReadonlyMap<string, Trait> = new Map(ALL_TRAITS.map((trait) => [normalizeKey(trait), trait]));

/** The trait a key names, or undefined when it names none. */
export function traitOf(key: string): Trait | undefined {
  return TRAIT_BY_KEY.get(normalizeKey(key));
}

export function isNumericTrait(trait: Trait): trait is NumericTrait {
  return trait in NUMERIC_TRAITS;
}

/** True for a non-empty string that fits the census category length. */
export function isCategoryText(value: unknown): value is string {
  return typeof value === 'string' && value.trim().length > 0 && value.trim().length <= MAX_CATEGORY_LENGTH;
}
