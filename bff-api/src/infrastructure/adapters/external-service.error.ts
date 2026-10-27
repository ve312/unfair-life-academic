export type ExternalServiceName = 'ai-service' | 'cms';

/**
 * An external dependency (the FastAPI AI service or the Strapi CMS) could not
 * be reached or answered in an unexpected way. The web layer maps it to a
 * 502/503 response; the message never contains credentials.
 */
export class ExternalServiceError extends Error {
  readonly service: ExternalServiceName;
  /** HTTP status returned by the service, when it answered at all. */
  readonly status: number | undefined;

  constructor(service: ExternalServiceName, message: string, options: { status?: number; cause?: unknown } = {}) {
    super(`[${service}] ${message}`, { cause: options.cause });
    this.name = 'ExternalServiceError';
    this.service = service;
    this.status = options.status;
  }
}
