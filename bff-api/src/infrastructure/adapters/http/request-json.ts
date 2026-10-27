import { ExternalServiceError, type ExternalServiceName } from '../external-service.error.js';

export interface JsonRequest {
  method: 'GET' | 'POST';
  headers?: Record<string, string>;
  body?: unknown;
  timeoutMs: number;
}

export interface JsonResponse {
  status: number;
  /** Parsed JSON body, or null when the response had no body. */
  body: unknown;
}

const MAX_ERROR_DETAIL_LENGTH = 300;

/**
 * Sends an HTTP request with the native fetch API and parses a JSON reply.
 *
 * Every transport problem (unreachable host, DNS failure, timeout, non-JSON
 * reply) becomes an ExternalServiceError. HTTP error statuses are returned to
 * the caller, which knows what a 404 or 422 means for its service.
 */
export async function requestJson(service: ExternalServiceName, url: string, request: JsonRequest): Promise<JsonResponse> {
  const target = describeTarget(url);
  let response: Response;
  try {
    response = await fetch(url, {
      method: request.method,
      headers: {
        Accept: 'application/json',
        ...(request.body === undefined ? {} : { 'Content-Type': 'application/json' }),
        ...request.headers,
      },
      body: request.body === undefined ? undefined : JSON.stringify(request.body),
      signal: AbortSignal.timeout(request.timeoutMs),
    });
  } catch (error) {
    if (error instanceof DOMException && (error.name === 'TimeoutError' || error.name === 'AbortError')) {
      throw new ExternalServiceError(service, `${request.method} ${target} timed out after ${request.timeoutMs} ms`, {
        cause: error,
      });
    }
    const reason = error instanceof Error ? (error.cause instanceof Error ? error.cause.message : error.message) : String(error);
    throw new ExternalServiceError(service, `${request.method} ${target} failed: ${reason}`, { cause: error });
  }

  let text: string;
  try {
    text = await response.text();
  } catch (error) {
    throw new ExternalServiceError(service, `${request.method} ${target} response could not be read`, {
      status: response.status,
      cause: error,
    });
  }
  if (text.trim().length === 0) {
    return { status: response.status, body: null };
  }
  try {
    return { status: response.status, body: JSON.parse(text) as unknown };
  } catch (error) {
    throw new ExternalServiceError(
      service,
      `${request.method} ${target} returned non-JSON content (status ${response.status})`,
      { status: response.status, cause: error },
    );
  }
}

/** Extracts a short, human-readable reason from an error response body. */
export function describeErrorBody(body: unknown): string {
  if (body === null || body === undefined) {
    return 'no response body';
  }
  if (typeof body === 'object') {
    const record = body as Record<string, unknown>;
    const nested = record.error as Record<string, unknown> | undefined;
    const candidate = typeof nested?.message === 'string' ? nested.message : record.detail;
    if (typeof candidate === 'string') {
      return truncate(candidate);
    }
    return truncate(JSON.stringify(body));
  }
  return truncate(typeof body === 'string' ? body : JSON.stringify(body));
}

function describeTarget(url: string): string {
  try {
    const parsed = new URL(url);
    return `${parsed.origin}${parsed.pathname}`;
  } catch {
    return url;
  }
}

function truncate(text: string): string {
  return text.length > MAX_ERROR_DETAIL_LENGTH ? `${text.slice(0, MAX_ERROR_DETAIL_LENGTH)}...` : text;
}
