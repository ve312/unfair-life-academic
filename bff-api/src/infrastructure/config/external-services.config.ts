import { Logger } from '@nestjs/common';

/**
 * Configuration for the two external services, read from environment
 * variables (see .env.example). Values are validated once at startup so a
 * misconfiguration fails fast instead of on the first player request.
 */

export const AI_SERVICE_CONFIG = Symbol('AiServiceConfig');
export const STRAPI_CONFIG = Symbol('StrapiConfig');

export interface AiServiceConfig {
  /** FastAPI origin, without a trailing slash, e.g. http://localhost:8000 */
  baseUrl: string;
  /** Sent as the X-BFF-Token header; must equal the AI service BFF_SECRET_TOKEN. */
  token: string;
  timeoutMs: number;
}

export interface StrapiConfig {
  /** Strapi origin the BFF calls, without a trailing slash, e.g. http://localhost:1337 */
  baseUrl: string;
  /** Public origin used to turn Strapi's relative media paths into URLs the game client can load. */
  mediaBaseUrl: string;
  /** Sent verbatim as the Authorization header; must equal the CMS BFF_SECRET_TOKEN. */
  token: string;
  timeoutMs: number;
}

type Env = Record<string, string | undefined>;

/** Mirrors the CMS policy's development fallback (cms/src/policies/is-bff-server.ts). */
export const DEV_FALLBACK_BFF_TOKEN = 'dev-secret-token';

const DEFAULT_AI_SERVICE_URL = 'http://localhost:8000';
const DEFAULT_STRAPI_URL = 'http://localhost:1337';
// 60 s: free-tier services sleep when idle and need 30-50 s to wake up (cold start).
const DEFAULT_AI_TIMEOUT_MS = 60_000;
const DEFAULT_STRAPI_TIMEOUT_MS = 60_000;

const logger = new Logger('ExternalServicesConfig');

export function loadAiServiceConfig(env: Env = process.env): AiServiceConfig {
  return {
    baseUrl: parseBaseUrl('AI_SERVICE_URL', env.AI_SERVICE_URL, DEFAULT_AI_SERVICE_URL),
    token: resolveBffToken(env),
    timeoutMs: parseTimeout('AI_SERVICE_TIMEOUT_MS', env.AI_SERVICE_TIMEOUT_MS, DEFAULT_AI_TIMEOUT_MS),
  };
}

export function loadStrapiConfig(env: Env = process.env): StrapiConfig {
  const baseUrl = parseBaseUrl('STRAPI_URL', env.STRAPI_URL, DEFAULT_STRAPI_URL);
  return {
    baseUrl,
    mediaBaseUrl: parseBaseUrl('STRAPI_MEDIA_BASE_URL', env.STRAPI_MEDIA_BASE_URL, baseUrl),
    token: resolveBffToken(env),
    timeoutMs: parseTimeout('STRAPI_TIMEOUT_MS', env.STRAPI_TIMEOUT_MS, DEFAULT_STRAPI_TIMEOUT_MS),
  };
}

function resolveBffToken(env: Env): string {
  const token = env.BFF_SECRET_TOKEN?.trim();
  if (token) {
    return token;
  }
  if (env.NODE_ENV === 'production') {
    throw new Error('BFF_SECRET_TOKEN must be set in production; the CMS and the AI service reject every request without it.');
  }
  logger.warn(`BFF_SECRET_TOKEN is not set; using the development fallback '${DEV_FALLBACK_BFF_TOKEN}'.`);
  return DEV_FALLBACK_BFF_TOKEN;
}

function parseBaseUrl(name: string, value: string | undefined, fallback: string): string {
  const raw = value?.trim() || fallback;
  let url: URL;
  try {
    url = new URL(raw);
  } catch {
    throw new Error(`${name} must be an absolute URL, got '${raw}'`);
  }
  if (url.protocol !== 'http:' && url.protocol !== 'https:') {
    throw new Error(`${name} must use http or https, got '${raw}'`);
  }
  return raw.replace(/\/+$/, '');
}

function parseTimeout(name: string, value: string | undefined, fallback: number): number {
  if (value === undefined || value.trim() === '') {
    return fallback;
  }
  const parsed = Number(value);
  if (!Number.isInteger(parsed) || parsed <= 0) {
    throw new Error(`${name} must be a positive integer number of milliseconds, got '${value}'`);
  }
  return parsed;
}
