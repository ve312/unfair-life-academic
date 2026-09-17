import type { Core } from '@strapi/strapi';

/**
 * Behind Render's HTTPS proxy:
 *   - PUBLIC_URL is the public address (e.g. https://unfair-life-cms.onrender.com), used
 *     for absolute URLs such as admin links.
 *   - proxy.koa trusts the proxy's X-Forwarded-* headers, so Strapi knows requests arrive
 *     over HTTPS and accepts the secure admin session cookies. Without it, logging in to
 *     the admin panel fails in production.
 * Render provides PORT; HOST must stay 0.0.0.0 there.
 */
const config = ({ env }: Core.Config.Shared.ConfigParams): Core.Config.Server => ({
  host: env('HOST', '0.0.0.0'),
  port: env.int('PORT', 1337),
  url: env('PUBLIC_URL', ''),
  proxy: { koa: env.bool('IS_PROXIED', env('NODE_ENV') === 'production') },
  app: {
    keys: env.array('APP_KEYS')!,
  },
  webhooks: {
    populateRelations: env.bool('WEBHOOKS_POPULATE_RELATIONS', false),
  },
});

export default config;
