/**
 * Keeps the admin panel's field help (label, description, placeholder) in sync
 * with the `config.metadatas` declared in each content-type schema.json.
 *
 * Why this is needed: Strapi stores the content-manager view configuration in
 * the database, and reads `config.metadatas` from the schema only when it
 * creates the configuration of a field for the first time. A field that
 * already exists in a database (for example hidden_modifiers, added before its
 * help text) would otherwise never show the help, and later edits to the help
 * text in the schema would never reach the panel. On every start, this step
 * brings the panel up to date with the schema, so that the schema stays the
 * single source of truth.
 *
 * Text that an editor customised through "Configure the view" is never
 * overwritten. A stored value is replaced only when it is still a default:
 *
 * - it is empty, or it is a label still equal to the attribute name (Strapi's
 *   defaults); or
 * - it is exactly the text this step applied last time, which is remembered
 *   in Strapi's core store. That text is now outdated, and nobody edited it.
 */
import type { Core } from '@strapi/strapi';

const HELP_KEYS = ['label', 'description', 'placeholder'] as const;
const STORE_KEY = 'applied';

/**
 * Help texts that earlier versions of the schemas declared, applied before the core-store record
 * existed. They are known to be ours, so they are replaced like any outdated applied text.
 */
const SUPERSEDED_HELP: ReadonlySet<string> = new Set([
  'Usa formato JSON válido. Para sumar o restar valores numéricos: {"capital-gain": 5000, "age": 2}. Para cambiar estados: '
  + '{"workclass": "Private"}. Solo usa las variables permitidas del modelo: age, education-num, hours-per-week, capital-gain, '
  + 'capital-loss (números enteros) y workclass, education, marital-status, occupation, relationship, race, sex, native-country '
  + '(texto). Requiere activar requires_mobility_check. Déjalo vacío si la decisión no cambia al personaje.',
]);

type HelpKey = (typeof HELP_KEYS)[number];
type EditHelp = Partial<Record<HelpKey, string>>;
type FieldMetadata = { edit?: EditHelp & Record<string, unknown> } & Record<string, unknown>;
type SchemaWithHelp = { config?: { metadatas?: Record<string, { edit?: EditHelp }> } };
/** The text last applied, indexed by '<uid>.<field>.<key>'. */
type AppliedHelp = Record<string, string>;

/** True when the stored value may be replaced by the schema's help. */
function isReplaceable(key: HelpKey, value: unknown, field: string, lastApplied: string | undefined): boolean {
  if (value === undefined || value === null || value === '') {
    return true;
  }
  if (key === 'label' && value === field) {
    return true;
  }
  return value === lastApplied || (typeof value === 'string' && SUPERSEDED_HELP.has(value));
}

/**
 * Applies the schema-declared field help to every api:: content type that declares some.
 *
 * @param strapi - the Strapi instance, called from bootstrap() once the content manager is ready.
 */
export async function syncFieldHelp(strapi: Core.Strapi): Promise<void> {
  const contentTypes = strapi.plugin('content-manager').service('content-types');
  const store = strapi.store({ type: 'core', name: 'field-help' });
  const applied = ((await store.get({ key: STORE_KEY })) as AppliedHelp | null) ?? {};
  let appliedChanged = false;

  for (const [uid, model] of Object.entries(strapi.contentTypes)) {
    const declared = (model as SchemaWithHelp).config?.metadatas;
    if (!uid.startsWith('api::') || declared === undefined) {
      continue;
    }
    const configuration = await contentTypes.findConfiguration({ uid });
    const metadatas = configuration.metadatas as Record<string, FieldMetadata> | undefined;
    if (metadatas === undefined) {
      continue;
    }

    const filled: string[] = [];
    for (const [field, help] of Object.entries(declared)) {
      const edit = metadatas[field]?.edit;
      if (edit === undefined || help.edit === undefined) {
        continue;
      }
      for (const key of HELP_KEYS) {
        const text = help.edit[key];
        const appliedKey = `${uid}.${field}.${key}`;
        if (text === undefined) {
          continue;
        }
        if (edit[key] !== text && isReplaceable(key, edit[key], field, applied[appliedKey])) {
          edit[key] = text;
          filled.push(`${field}.${key}`);
        }
        // Remember the schema text whenever the panel shows it, so that a later schema edit can replace it.
        if (edit[key] === text && applied[appliedKey] !== text) {
          applied[appliedKey] = text;
          appliedChanged = true;
        }
      }
    }

    if (filled.length > 0) {
      await contentTypes.updateConfiguration({ uid }, configuration);
      strapi.log.info(`[field-help] ${uid}: updated ${filled.join(', ')}`);
    }
  }

  if (appliedChanged) {
    await store.set({ key: STORE_KEY, value: applied });
  }
}
