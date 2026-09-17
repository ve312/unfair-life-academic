import type { Core } from '@strapi/strapi';
import { syncFieldHelp } from './admin-field-help';
import { seedTurnoDeNoche } from './seeds/turno-de-noche';

export default {
  /**
   * An asynchronous register function that runs before
   * your application is initialized.
   *
   * This gives you an opportunity to extend code.
   */
  register(/* { strapi }: { strapi: Core.Strapi } */) {},

  /**
   * An asynchronous bootstrap function that runs before
   * your application gets started.
   *
   * This gives you an opportunity to set up your data model,
   * run jobs, or perform some special logic.
   */
  async bootstrap({ strapi }: { strapi: Core.Strapi }) {
    // Show the schema-declared help (e.g. how to write hidden_modifiers) in the admin panel.
    await syncFieldHelp(strapi);

    // Idempotent story seeds (they create a story only when it does not exist). A failure is logged,
    // never fatal: the CMS must start even if a seed cannot run. SEED_STORIES=false disables them.
    if (process.env.SEED_STORIES !== 'false') {
      try {
        await seedTurnoDeNoche(strapi);
      } catch (error) {
        strapi.log.error(`[seed] "Turno de Noche" failed and was rolled back: ${error instanceof Error ? error.message : String(error)}`);
      }
    }
  },
};
