/**
 * Global policy `global::is-bff-server`.
 *
 * Allows a request only when its `Authorization` header exactly matches the
 * BFF_SECRET_TOKEN environment variable, i.e. the request comes from the
 * NestJS BFF. Any other caller receives 403 Forbidden.
 *
 * Fallback: when BFF_SECRET_TOKEN is not set, 'dev-secret-token' is accepted
 * in development only. In production a missing token fails closed (every
 * request is rejected) so a publicly known default can never unlock the API.
 *
 * On success the policy also records the BFF as the authenticated caller
 * (ctx.state.auth) with a read-only scope verifier. The routes run with
 * Strapi's own auth disabled, and without an auth state Strapi's output
 * sanitizer silently strips every relation from the response. The verifier
 * grants exactly the scopes needed to populate the story graph and nothing
 * else.
 */
import { createHash, timingSafeEqual } from 'node:crypto';
import type { Core } from '@strapi/strapi';

const DEV_FALLBACK_TOKEN = 'dev-secret-token';

const READABLE_UIDS = [
  'api::story.story',
  'api::scene.scene',
  'api::decision.decision',
  'api::audio-track.audio-track',
  'plugin::upload.file',
] as const;
const READ_ACTIONS = ['find', 'findOne'] as const;
const ALLOWED_SCOPES = new Set<string>(
  READABLE_UIDS.flatMap((uid) => READ_ACTIONS.map((action) => `${uid}.${action}`)),
);

const bffStrategy = {
  name: 'bff-server',
  verify(_auth: unknown, config: { scope?: string | string[] } = {}): void {
    const scopes = config.scope === undefined ? [] : [config.scope].flat();
    const denied = scopes.find((scope) => !ALLOWED_SCOPES.has(scope));
    if (denied !== undefined) {
      throw new Error(`BFF is not allowed to access scope '${denied}'`);
    }
  },
};

let warnedAboutFallback = false;

const resolveExpectedToken = (strapi: Core.Strapi): string | null => {
  const configured = process.env.BFF_SECRET_TOKEN?.trim();
  if (configured) {
    return configured;
  }
  if (process.env.NODE_ENV === 'production') {
    strapi.log.error('[is-bff-server] BFF_SECRET_TOKEN is not set in production; rejecting all requests.');
    return null;
  }
  if (!warnedAboutFallback) {
    strapi.log.warn(`[is-bff-server] BFF_SECRET_TOKEN is not set; using the development fallback '${DEV_FALLBACK_TOKEN}'.`);
    warnedAboutFallback = true;
  }
  return DEV_FALLBACK_TOKEN;
};

// Hashing both values gives equal-length buffers, so the comparison takes the
// same time regardless of where (or whether) the strings differ.
const constantTimeEquals = (received: string, expected: string): boolean =>
  timingSafeEqual(
    createHash('sha256').update(received).digest(),
    createHash('sha256').update(expected).digest(),
  );

const isBffServer: Core.PolicyHandler = (policyContext, _config, { strapi }) => {
  const expected = resolveExpectedToken(strapi);
  const received = policyContext.request.headers.authorization;

  if (expected === null || typeof received !== 'string' || !constantTimeEquals(received, expected)) {
    strapi.log.warn(
      `[is-bff-server] Rejected ${policyContext.request.method} ${policyContext.request.url} from ${policyContext.request.ip}`,
    );
    return false;
  }

  // The policy context is a shallow copy of the Koa context, so `state` is the
  // same object the controller reads; Strapi's PolicyContext type omits it.
  const { state } = policyContext as typeof policyContext & { state: Record<string, unknown> };
  state.isAuthenticated = true;
  state.auth = {
    strategy: bffStrategy,
    credentials: { client: 'nestjs-bff' },
  };
  return true;
};

export default isBffServer;
