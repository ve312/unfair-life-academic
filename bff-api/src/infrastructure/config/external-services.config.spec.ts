import { DEV_FALLBACK_BFF_TOKEN, loadAiServiceConfig, loadStrapiConfig } from './external-services.config.js';

describe('external services config', () => {
  it('uses local defaults for development', () => {
    expect(loadAiServiceConfig({})).toEqual({ baseUrl: 'http://localhost:8000', token: DEV_FALLBACK_BFF_TOKEN, timeoutMs: 60_000 });
    expect(loadStrapiConfig({})).toEqual({
      baseUrl: 'http://localhost:1337',
      mediaBaseUrl: 'http://localhost:1337',
      token: DEV_FALLBACK_BFF_TOKEN,
      timeoutMs: 60_000,
    });
  });

  it('reads and normalises environment variables', () => {
    const config = loadStrapiConfig({
      STRAPI_URL: 'http://cms:1337/',
      STRAPI_MEDIA_BASE_URL: 'https://media.example.com//',
      BFF_SECRET_TOKEN: '  abc123  ',
      STRAPI_TIMEOUT_MS: '2500',
    });

    expect(config).toEqual({ baseUrl: 'http://cms:1337', mediaBaseUrl: 'https://media.example.com', token: 'abc123', timeoutMs: 2500 });
  });

  it('sends the shared BFF_SECRET_TOKEN to the AI service too', () => {
    expect(loadAiServiceConfig({ BFF_SECRET_TOKEN: ' abc123 ' }).token).toBe('abc123');
  });

  it('refuses to start in production without BFF_SECRET_TOKEN', () => {
    expect(() => loadStrapiConfig({ NODE_ENV: 'production' })).toThrow('BFF_SECRET_TOKEN must be set in production');
    expect(() => loadAiServiceConfig({ NODE_ENV: 'production' })).toThrow('BFF_SECRET_TOKEN must be set in production');
  });

  it.each([
    ['a relative URL', { AI_SERVICE_URL: 'localhost:8000' }],
    ['a non-http URL', { AI_SERVICE_URL: 'ftp://ai:8000' }],
    ['a zero timeout', { AI_SERVICE_TIMEOUT_MS: '0' }],
    ['a non-numeric timeout', { AI_SERVICE_TIMEOUT_MS: 'fast' }],
  ])('rejects %s', (_label, env) => {
    expect(() => loadAiServiceConfig(env)).toThrow();
  });
});
