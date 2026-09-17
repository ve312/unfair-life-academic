/**
 * Validation of a decision's hidden modifiers: the narrative context that the
 * BFF injects into the character before the AI evaluates a mobility check.
 *
 * Mirrors the BFF rule (bff-api/src/core/domain/entities/hidden-modifiers.entity.ts),
 * so that anything the admin panel accepts can always be played:
 *
 * - Empty (null, absent or {}) means "no modifiers".
 * - Otherwise the value must be a JSON object whose keys name a modifiable
 *   trait, in any spelling ('capital-gain', 'capital_gain' or 'capitalGain').
 * - A numeric trait takes an integer, which is added (positive) or subtracted
 *   (negative). A categorical trait takes a non-empty string, which replaces
 *   the current value.
 * - A delta of 0 is the identity: adding zero changes nothing. This is what
 *   lets every new decision start from the template
 *   {"capital-gain": 0, "capital-loss": 0, "age": 0} without side effects.
 * - fnlwgt is not modifiable: it is a census sampling weight, not a
 *   circumstance that a story can change.
 * - Modifiers that change something only take effect through a mobility
 *   check, so a decision that declares them must have
 *   requires_mobility_check enabled.
 *
 * Messages are in Spanish because they are shown to content editors in the
 * Strapi admin panel.
 */
import { ALL_TRAITS, isCategoryText, isNumericTrait, MAX_CATEGORY_LENGTH, normalizeKey, traitOf, type Trait } from '../../../../character-traits';

export const HIDDEN_MODIFIERS_FIELD = 'hidden_modifiers';
export const MOBILITY_CHECK_FIELD = 'requires_mobility_check';

export interface ModifierIssue {
  path: string;
  message: string;
}

const ALLOWED_TRAITS = ALL_TRAITS.join(', ');

function isEmpty(value: unknown): boolean {
  return (
    value === null ||
    value === undefined ||
    (typeof value === 'object' && !Array.isArray(value) && Object.keys(value as object).length === 0)
  );
}

/**
 * Returns every problem with a decision's hidden modifiers (empty when they are valid).
 *
 * @param modifiers - the value of the hidden_modifiers field (any JSON value).
 * @param requiresMobilityCheck - the effective value of requires_mobility_check after the write.
 */
export function findModifierIssues(modifiers: unknown, requiresMobilityCheck: boolean): ModifierIssue[] {
  if (isEmpty(modifiers)) {
    return [];
  }
  const issue = (message: string): ModifierIssue => ({ path: HIDDEN_MODIFIERS_FIELD, message });

  if (typeof modifiers === 'string') {
    try {
      JSON.parse(modifiers);
    } catch {
      return [issue('Modificadores ocultos: el contenido no es un JSON válido.')];
    }
    return [issue(
      'Modificadores ocultos: debe ser un objeto JSON, no un texto entre comillas. '
      + 'Ejemplo: {"capital-gain": 5000, "workclass": "Private"}.',
    )];
  }
  if (typeof modifiers !== 'object' || Array.isArray(modifiers)) {
    return [issue('Modificadores ocultos: debe ser un objeto JSON, por ejemplo {"capital-gain": 5000, "age": 2}.')];
  }

  const issues: ModifierIssue[] = [];
  const keyBySeenTrait = new Map<Trait, string>();
  let changesSomething = false;
  for (const [key, value] of Object.entries(modifiers as Record<string, unknown>)) {
    if (normalizeKey(key) === 'fnlwgt') {
      issues.push(issue(
        `Modificadores ocultos: "${key}" no se puede modificar; es un peso muestral del censo, no una circunstancia del personaje.`,
      ));
      continue;
    }
    const trait = traitOf(key);
    if (trait === undefined) {
      issues.push(issue(`Modificadores ocultos: "${key}" no es una característica modificable. Permitidas: ${ALLOWED_TRAITS}.`));
      continue;
    }
    const previous = keyBySeenTrait.get(trait);
    if (previous !== undefined) {
      issues.push(issue(`Modificadores ocultos: "${key}" y "${previous}" modifican la misma característica (${trait}).`));
      continue;
    }
    keyBySeenTrait.set(trait, key);

    if (isNumericTrait(trait)) {
      if (typeof value !== 'number' || !Number.isSafeInteger(value)) {
        issues.push(issue(
          `Modificadores ocultos: "${key}" es numérica, así que su valor debe ser un número entero que se suma `
          + `(o se resta si es negativo); se recibió ${JSON.stringify(value)}.`,
        ));
      } else if (value !== 0) {
        changesSomething = true;
      }
    } else if (!isCategoryText(value)) {
      issues.push(issue(
        `Modificadores ocultos: "${key}" es categórica, así que su valor debe ser un texto no vacío de hasta `
        + `${MAX_CATEGORY_LENGTH} caracteres que reemplaza al actual; se recibió ${JSON.stringify(value)}.`,
      ));
    } else {
      changesSomething = true;
    }
  }

  if (changesSomething && !requiresMobilityCheck) {
    issues.push(issue(
      'Modificadores ocultos: solo se aplican en una prueba de movilidad. Activa "requires_mobility_check", '
      + 'deja los valores en 0 o vacía el campo.',
    ));
  }
  return issues;
}
