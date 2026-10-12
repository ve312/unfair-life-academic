type Env = Record<string, string | undefined>;

export interface HttpConfig {
  port: number;
  /** Always all interfaces: hosting platforms (Render) route traffic to the container's public interface. */
  host: string;
  /** Browser origins allowed to call the BFF. */
  corsOrigins: string[];
}

const DEFAULT_PORT = 3000;
const BIND_ALL_INTERFACES = '0.0.0.0';

/** The Vite dev server and `vite preview`, on both local host names. */
export const LOCAL_DEV_ORIGINS = Object.freeze([
  'http://localhost:5173',
  'http://127.0.0.1:5173',
  'http://localhost:4173',
  'http://127.0.0.1:4173',
]);

/**
 * Reads PORT (Render injects it; 3000 locally) and the allowed browser origins:
 *   FRONTEND_URL  the deployed game client, e.g. https://unfair-life.vercel.app
 *   CORS_ORIGINS  optional extra origins, comma-separated
 * Outside production the local Vite origins are always allowed too. Production
 * refuses to start without FRONTEND_URL, so CORS can never ship wide open or empty.
 */
export function loadHttpConfig(env: Env = process.env): HttpConfig {
  return { port: parsePort(env.PORT), host: BIND_ALL_INTERFACES, corsOrigins: resolveCorsOrigins(env) };
}

type CorsOriginCallback = (error: Error | null, allow?: boolean) => void;

/**
 * Builds the `origin` option of enableCors() for an exact allow-list. A rejected
 * origin still gets its response (the browser, not the server, blocks it), so
 * `onRejected` is called once per origin to make the misconfiguration visible in
 * the logs. Requests without an Origin header (curl, health probes) are allowed.
 */
export function createCorsOriginCheck(
  allowedOrigins: readonly string[],
  onRejected: (origin: string) => void,
): (origin: string | undefined, callback: CorsOriginCallback) => void {
  const allowed = new Set(allowedOrigins);
  const reported = new Set<string>();
  return (origin, callback) => {
    if (origin === undefined || allowed.has(origin)) {
      callback(null, true);
      return;
    }
    if (!reported.has(origin)) {
      reported.add(origin);
      onRejected(origin);
    }
    callback(null, false);
  };
}

function parsePort(value: string | undefined): number {
  if (value === undefined || value.trim() === '') {
    return DEFAULT_PORT;
  }
  const port = Number(value);
  if (!Number.isInteger(port) || port < 1 || port > 65535) {
    throw new Error(`PORT must be an integer between 1 and 65535, got '${value}'`);
  }
  return port;
}

function resolveCorsOrigins(env: Env): string[] {
  const isProduction = env.NODE_ENV === 'production';
  const frontend = parseOrigins('FRONTEND_URL', env.FRONTEND_URL);
  if (isProduction && frontend.length === 0) {
    throw new Error('FRONTEND_URL must be set in production to the game client origin (e.g. https://unfair-life.vercel.app)');
  }
  const origins = [...frontend, ...parseOrigins('CORS_ORIGINS', env.CORS_ORIGINS), ...(isProduction ? [] : LOCAL_DEV_ORIGINS)];
  return [...new Set(origins)];
}

/** Comma-separated origins; a trailing slash is tolerated (as copied from a browser), a path is not. */
function parseOrigins(name: string, value: string | undefined): string[] {
  const origins = (value ?? '')
    .split(',')
    .map((origin) => origin.trim().replace(/\/+$/, ''))
    .filter((origin) => origin.length > 0);
  for (const origin of origins) {
    let url: URL;
    try {
      url = new URL(origin);
    } catch {
      throw new Error(`${name} contains an invalid origin: '${origin}'`);
    }
    if (url.origin !== origin) {
      throw new Error(`${name} entries must be bare origins like 'https://unfair-life.vercel.app', got '${origin}'`);
    }
  }
  return origins;
}
