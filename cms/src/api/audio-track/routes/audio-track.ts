/**
 * Content API routes for api::audio-track.audio-track.
 *
 * Read-only and BFF-only: just `find` and `findOne` are exposed, guarded by
 * the `global::is-bff-server` policy. Strapi's own authentication is disabled
 * on them (`auth: false`) because it would otherwise try to parse the
 * Authorization header as a JWT/API token and reject the BFF secret with 401
 * before the policy runs. On success the policy registers the BFF as a
 * read-only caller so relations and media can still be populated. Editors
 * manage content through the admin panel, which uses its own authenticated API.
 */
import { factories } from '@strapi/strapi';

export default factories.createCoreRouter('api::audio-track.audio-track', {
  only: ['find', 'findOne'],
  config: {
    find: {
      auth: false,
      policies: ['global::is-bff-server'],
    },
    findOne: {
      auth: false,
      policies: ['global::is-bff-server'],
    },
  },
});
