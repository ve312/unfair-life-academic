import { createCorsOriginCheck, LOCAL_DEV_ORIGINS, loadHttpConfig } from './http.config.js';

describe('http config', () => {
  it('defaults to port 3000 on all interfaces, allowing the local Vite origins in development', () => {
    expect(loadHttpConfig({})).toEqual({ port: 3000, host: '0.0.0.0', corsOrigins: [...LOCAL_DEV_ORIGINS] });
  });

  it('binds to the PORT the platform injects', () => {
    expect(loadHttpConfig({ PORT: '10000' }).port).toBe(10000);
  });

  it('allows FRONTEND_URL alongside localhost in development', () => {
    const { corsOrigins } = loadHttpConfig({ FRONTEND_URL: 'https://unfair-life.vercel.app' });

    expect(corsOrigins[0]).toBe('https://unfair-life.vercel.app');
    expect(corsOrigins).toEqual(expect.arrayContaining(['http://localhost:5173', 'http://127.0.0.1:5173']));
  });

  it('allows only FRONTEND_URL (and CORS_ORIGINS) in production', () => {
    const { corsOrigins } = loadHttpConfig({
      NODE_ENV: 'production',
      FRONTEND_URL: 'https://unfair-life.vercel.app/',
      CORS_ORIGINS: ' https://staging.unfair-life.example , https://unfair-life.vercel.app ',
    });

    expect(corsOrigins).toEqual(['https://unfair-life.vercel.app', 'https://staging.unfair-life.example']);
  });

  it('refuses to start in production without FRONTEND_URL', () => {
    expect(() => loadHttpConfig({ NODE_ENV: 'production' })).toThrow('FRONTEND_URL must be set in production');
    expect(() => loadHttpConfig({ NODE_ENV: 'production', CORS_ORIGINS: 'https://x.example' })).toThrow('FRONTEND_URL');
  });

  it.each([
    ['an invalid port', { PORT: '70000' }],
    ['a non-numeric port', { PORT: 'http' }],
    ['a FRONTEND_URL with a path', { FRONTEND_URL: 'https://unfair-life.vercel.app/game' }],
    ['a FRONTEND_URL without a scheme', { FRONTEND_URL: 'unfair-life.vercel.app' }],
    ['a malformed CORS_ORIGINS entry', { CORS_ORIGINS: 'localhost' }],
  ])('rejects %s', (_label, env) => {
    expect(() => loadHttpConfig(env)).toThrow();
  });
});

describe('createCorsOriginCheck', () => {
  const PRODUCTION = 'https://unfair-life.vercel.app';
  const BRANCH_ALIAS = 'https://unfair-life-git-main-team.vercel.app';

  function decide(check: ReturnType<typeof createCorsOriginCheck>, origin: string | undefined): boolean | undefined {
    let allowed: boolean | undefined;
    check(origin, (_error, allow) => {
      allowed = allow;
    });
    return allowed;
  }

  it('allows listed origins and requests without an Origin header', () => {
    const check = createCorsOriginCheck([PRODUCTION], vi.fn());

    expect(decide(check, PRODUCTION)).toBe(true);
    expect(decide(check, undefined)).toBe(true);
  });

  it('rejects an unlisted origin, such as a Vercel branch alias, and reports it once', () => {
    const onRejected = vi.fn();
    const check = createCorsOriginCheck([PRODUCTION], onRejected);

    expect(decide(check, BRANCH_ALIAS)).toBe(false);
    expect(decide(check, BRANCH_ALIAS)).toBe(false);
    expect(onRejected).toHaveBeenCalledTimes(1);
    expect(onRejected).toHaveBeenCalledWith(BRANCH_ALIAS);
  });
});
