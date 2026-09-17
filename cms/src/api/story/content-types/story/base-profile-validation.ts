/**
 * Validation of a story's base profile: the character every playthrough of
 * the story starts as, and on which the decisions' hidden modifiers act.
 *
 * Mirrors the BFF parser (bff-api/src/core/domain/entities/base-profile.ts),
 * so that anything the admin panel accepts can always be played:
 *
 * - The profile is optional. Purely narrative stories never consult the AI,
 *   so an empty value (null, absent or {}) is valid and means "no profile".
 *   The BFF requires a profile only when a decision triggers a mobility check.
 * - A non-empty value must be a JSON object holding exactly the 13 traits of the
 *   model, in any spelling ('capital-gain', 'capital_gain' or 'capitalGain').
 * - Numeric traits are integers inside the ranges the model saw in training.
 *   Categorical traits are non-empty census spellings.
 * - fnlwgt is not part of a profile: it is a census sampling weight.
 *
 * It also enforces one rule of internal coherence that the BFF does not need
 * for safety but the model needs for meaning. In the census, education and
 * education-num encode the same fact (Bachelors is always 13, HS-grad always
 * 9). A profile in which they disagree describes a person absent from the
 * training data, so the pair must match whenever education is a census level.
 *
 * Messages are in Spanish because they are shown to content editors in the
 * Strapi admin panel.
 */
import {
  ALL_TRAITS,
  isCategoryText,
  isNumericTrait,
  MAX_CATEGORY_LENGTH,
  normalizeKey,
  NUMERIC_TRAITS,
  traitOf,
  type Trait,
} from '../../../../character-traits';

export const BASE_PROFILE_FIELD = 'base_profile';

/** Census education levels and their ordinal years of schooling (education-num). */
export const EDUCATION_NUM_BY_LEVEL: Readonly<Record<string, number>> = {
  Preschool: 1,
  '1st-4th': 2,
  '5th-6th': 3,
  '7th-8th': 4,
  '9th': 5,
  '10th': 6,
  '11th': 7,
  '12th': 8,
  'HS-grad': 9,
  'Some-college': 10,
  'Assoc-voc': 11,
  'Assoc-acdm': 12,
  Bachelors: 13,
  Masters: 14,
  'Prof-school': 15,
  Doctorate: 16,
};

export interface ProfileIssue {
  path: string;
  message: string;
}

/** Spelling used in messages: the kebab-case form editors see in the default profile. */
function displayName(trait: Trait): string {
  return trait.replace(/[A-Z]/g, (letter) => `-${letter.toLowerCase()}`);
}

/**
 * Returns every problem with a story's base profile (empty when it is valid).
 *
 * @param profile - the value of the base_profile field (any JSON value).
 */
export function findProfileIssues(profile: unknown): ProfileIssue[] {
  const issue = (message: string): ProfileIssue => ({ path: BASE_PROFILE_FIELD, message });

  const isEmptyObject =
    typeof profile === 'object' && profile !== null && !Array.isArray(profile) && Object.keys(profile).length === 0;
  if (profile === null || profile === undefined || isEmptyObject) {
    return []; // No profile: valid for a story without mobility checks.
  }
  if (typeof profile === 'string') {
    return [issue('Perfil base: debe ser un objeto JSON, no un texto entre comillas.')];
  }
  if (typeof profile !== 'object' || Array.isArray(profile)) {
    return [issue('Perfil base: debe ser un objeto JSON con las 13 características del personaje.')];
  }

  const issues: ProfileIssue[] = [];
  const values = new Map<Trait, unknown>();
  const keyBySeenTrait = new Map<Trait, string>();
  for (const [key, value] of Object.entries(profile as Record<string, unknown>)) {
    if (normalizeKey(key) === 'fnlwgt') {
      issues.push(issue(`Perfil base: "${key}" no forma parte del perfil; es un peso muestral del censo, no una característica del personaje.`));
      continue;
    }
    const trait = traitOf(key);
    if (trait === undefined) {
      issues.push(issue(`Perfil base: "${key}" no es una característica del modelo. Permitidas: ${ALL_TRAITS.map(displayName).join(', ')}.`));
      continue;
    }
    const previous = keyBySeenTrait.get(trait);
    if (previous !== undefined) {
      issues.push(issue(`Perfil base: "${key}" y "${previous}" definen la misma característica (${displayName(trait)}).`));
      continue;
    }
    keyBySeenTrait.set(trait, key);
    values.set(trait, value);

    if (isNumericTrait(trait)) {
      const { min, max } = NUMERIC_TRAITS[trait];
      if (typeof value !== 'number' || !Number.isSafeInteger(value)) {
        issues.push(issue(`Perfil base: "${key}" debe ser un número entero; se recibió ${JSON.stringify(value)}.`));
      } else if (value < min || value > max) {
        const range = max === Number.MAX_SAFE_INTEGER ? `mayor o igual que ${min}` : `entre ${min} y ${max}`;
        issues.push(issue(`Perfil base: "${key}" debe estar ${range} (el rango que vio el modelo); se recibió ${value}.`));
      }
    } else if (!isCategoryText(value)) {
      issues.push(issue(
        `Perfil base: "${key}" debe ser un texto no vacío de hasta ${MAX_CATEGORY_LENGTH} caracteres; se recibió ${JSON.stringify(value)}.`,
      ));
    }
  }

  const missing = ALL_TRAITS.filter((trait) => !values.has(trait));
  if (missing.length > 0) {
    issues.push(issue(`Perfil base: faltan características: ${missing.map(displayName).join(', ')}.`));
  }

  const education = values.get('education');
  const educationNum = values.get('educationNum');
  const expected = typeof education === 'string' ? EDUCATION_NUM_BY_LEVEL[education.trim()] : undefined;
  if (expected !== undefined && typeof educationNum === 'number' && educationNum !== expected) {
    issues.push(issue(
      `Perfil base: "education" es "${(education as string).trim()}", que en el censo corresponde a "education-num": ${expected}; `
      + `se recibió ${educationNum}. Ambos campos describen el mismo nivel educativo y deben coincidir.`,
    ));
  }
  return issues;
}
