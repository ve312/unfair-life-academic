import type { Mock } from 'vitest';
import { InvalidStoryContentError, StoryNodeNotFoundError, StoryNotFoundError } from '../../../core/domain/errors/domain.error.js';
import type { StrapiConfig } from '../../config/external-services.config.js';
import { ExternalServiceError } from '../external-service.error.js';
import { StrapiContentAdapter, toStrapiQueryString } from './strapi-content.adapter.js';

const config: StrapiConfig = {
  baseUrl: 'http://cms.internal:1337',
  mediaBaseUrl: 'https://media.unfair-life.example',
  token: 'secret-bff-token',
  timeoutMs: 1000,
};

const scene = {
  documentId: 'intro-doc',
  title: 'Intro',
  timer_duration: 10,
  is_cinematic: false,
  background_media: null,
  audio_track: { loop: true, default_volume: 0.6, audio_file: { url: '/uploads/rain.mp3' } },
  available_decisions: [
    { documentId: 'work-doc', text: 'Start working', requires_mobility_check: false, next_scene: { documentId: 'factory-doc' }, failure_scene: null },
  ],
};

const jsonResponse = (status: number, body: unknown): Response =>
  new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } });

describe('StrapiContentAdapter', () => {
  let fetchMock: Mock<typeof fetch>;
  let adapter: StrapiContentAdapter;

  beforeEach(() => {
    fetchMock = vi.fn<typeof fetch>();
    vi.stubGlobal('fetch', fetchMock);
    adapter = new StrapiContentAdapter(config);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  const lastRequest = (): { url: URL; init: RequestInit } => {
    const [input, init] = fetchMock.mock.calls.at(-1) as [string, RequestInit];
    return { url: new URL(input), init };
  };

  it('fetches a scene with the bare BFF token and the explicit populate', async () => {
    fetchMock.mockResolvedValue(jsonResponse(200, { data: scene, meta: {} }));

    const node = await adapter.getStoryNode('intro-doc');

    const { url, init } = lastRequest();
    expect(url.origin + url.pathname).toBe('http://cms.internal:1337/api/scenes/intro-doc');
    expect((init.headers as Record<string, string>).Authorization).toBe('secret-bff-token');
    expect(init.method).toBe('GET');
    expect(url.searchParams.get('populate[available_decisions][populate][failure_scene][fields][0]')).toBe('title');
    expect(url.searchParams.get('populate[audio_track][populate][audio_file][fields][0]')).toBe('url');
    expect(url.searchParams.get('populate[available_decisions][fields][2]')).toBe('hidden_modifiers');
    expect(url.searchParams.get('fields[1]')).toBe('context');
    expect(url.searchParams.get('populate[story][fields][1]')).toBe('base_profile');
    const sceneFields = [...url.searchParams.entries()].filter(([k]) => /^fields\[\d+\]$/.test(k)).map(([, v]) => v);
    expect(sceneFields).toEqual(expect.arrayContaining(['media_start_time', 'media_end_time', 'audio_start_time', 'audio_end_time']));
    expect(node.audio?.url).toBe('https://media.unfair-life.example/uploads/rain.mp3');
    expect(node.findDecision('work-doc')?.nextSceneId).toBe('factory-doc');
  });

  it("fetches a story and maps its starting scene", async () => {
    fetchMock.mockResolvedValue(jsonResponse(200, { data: { documentId: 'story-doc', title: 'Born in Guatemala', starting_scene: scene }, meta: {} }));

    const node = await adapter.getStartingNode('story-doc');

    const { url } = lastRequest();
    expect(url.pathname).toBe('/api/stories/story-doc');
    expect(url.searchParams.get('populate[starting_scene][populate][available_decisions][fields][0]')).toBe('text');
    expect(node.id).toBe('intro-doc');
  });

  it('URL-encodes document ids', async () => {
    fetchMock.mockResolvedValue(jsonResponse(200, { data: scene, meta: {} }));

    await adapter.getStoryNode('../admin');

    expect(fetchMock.mock.calls[0]?.[0]).toContain('/api/scenes/..%2Fadmin?');
  });

  it('maps a 404 to StoryNodeNotFoundError / StoryNotFoundError', async () => {
    fetchMock.mockImplementation(async () => jsonResponse(404, { data: null, error: { status: 404, name: 'NotFoundError', message: 'Not Found' } }));

    await expect(adapter.getStoryNode('missing')).rejects.toBeInstanceOf(StoryNodeNotFoundError);
    await expect(adapter.getStartingNode('missing')).rejects.toBeInstanceOf(StoryNotFoundError);
  });

  it('rejects blank ids without calling the CMS', async () => {
    await expect(adapter.getStoryNode(' ')).rejects.toBeInstanceOf(StoryNodeNotFoundError);
    await expect(adapter.getStartingNode('')).rejects.toBeInstanceOf(StoryNotFoundError);
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it('reports a rejected token as an ExternalServiceError without leaking it', async () => {
    fetchMock.mockResolvedValue(jsonResponse(403, { data: null, error: { status: 403, name: 'PolicyError', message: 'Policy Failed' } }));

    const error = await adapter.getStoryNode('intro-doc').catch((caught: unknown) => caught);

    expect(error).toBeInstanceOf(ExternalServiceError);
    expect(error).toMatchObject({ service: 'cms', status: 403 });
    expect((error as Error).message).not.toContain('secret-bff-token');
  });

  it('reports server errors with the CMS message', async () => {
    fetchMock.mockResolvedValue(jsonResponse(500, { data: null, error: { status: 500, message: 'Internal Server Error' } }));

    await expect(adapter.getStoryNode('intro-doc')).rejects.toThrow('[cms] GET /api/scenes/intro-doc responded 500: Internal Server Error');
  });

  it('wraps network failures in ExternalServiceError', async () => {
    fetchMock.mockRejectedValue(new TypeError('fetch failed', { cause: new Error('connect ECONNREFUSED 127.0.0.1:1337') }));

    await expect(adapter.getStoryNode('intro-doc')).rejects.toThrow(/\[cms\] GET .* failed: connect ECONNREFUSED/);
  });

  it('wraps timeouts in ExternalServiceError', async () => {
    fetchMock.mockRejectedValue(new DOMException('The operation was aborted due to timeout', 'TimeoutError'));

    await expect(adapter.getStoryNode('intro-doc')).rejects.toThrow('timed out after 1000 ms');
  });

  it('surfaces malformed content as InvalidStoryContentError', async () => {
    fetchMock.mockResolvedValue(jsonResponse(200, { data: { ...scene, available_decisions: undefined }, meta: {} }));

    await expect(adapter.getStoryNode('intro-doc')).rejects.toBeInstanceOf(InvalidStoryContentError);
  });
});

describe('toStrapiQueryString', () => {
  it('serialises nested objects and arrays into bracket notation', () => {
    const query = decodeURIComponent(toStrapiQueryString({ fields: ['title'], populate: { audio_track: { fields: ['loop', 'url'] } } }));

    expect(query).toBe('fields[0]=title&populate[audio_track][fields][0]=loop&populate[audio_track][fields][1]=url');
  });
});

describe('StrapiContentAdapter.listStories', () => {
  let fetchMock: Mock<typeof fetch>;
  let adapter: StrapiContentAdapter;

  const listResponse = (data: unknown[], page = 1, pageCount = 1): Response =>
    new Response(JSON.stringify({ data, meta: { pagination: { page, pageSize: 100, pageCount, total: data.length } } }), {
      status: 200,
      headers: { 'Content-Type': 'application/json' },
    });
  const story = (n: number, extra: Record<string, unknown> = {}) => ({
    id: n, documentId: `story-${n}`, title: `Story ${n}`, short_description: `About story ${n}`, year: 2026,
    cover_image: { id: n, documentId: `img-${n}`, url: `/uploads/cover_${n}.jpg` }, ...extra,
  });

  beforeEach(() => {
    fetchMock = vi.fn<typeof fetch>();
    vi.stubGlobal('fetch', fetchMock);
    adapter = new StrapiContentAdapter(config);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('requests only playable stories, sorted, with the menu fields and the bare BFF token', async () => {
    fetchMock.mockResolvedValue(listResponse([story(1)]));

    const stories = await adapter.listStories();

    const [input, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    const url = new URL(input);
    expect(url.origin + url.pathname).toBe('http://cms.internal:1337/api/stories');
    expect((init.headers as Record<string, string>).Authorization).toBe('secret-bff-token');
    expect(url.searchParams.get('filters[starting_scene][id][$notNull]')).toBe('true');
    expect(url.searchParams.get('sort[0]')).toBe('title:asc');
    expect(url.searchParams.getAll('fields[0]').concat(url.searchParams.getAll('fields[1]'), url.searchParams.getAll('fields[2]')))
      .toEqual(['title', 'short_description', 'year']);
    expect(url.searchParams.get('populate[cover_image][fields][0]')).toBe('url');
    expect(url.searchParams.get('pagination[page]')).toBe('1');
    expect(url.searchParams.get('pagination[pageSize]')).toBe('100');
    expect(stories).toEqual([
      expect.objectContaining({
        id: 'story-1', title: 'Story 1', shortDescription: 'About story 1', year: 2026,
        coverImageUrl: 'https://media.unfair-life.example/uploads/cover_1.jpg',
      }),
    ]);
  });

  it('follows pagination until the last page', async () => {
    fetchMock
      .mockResolvedValueOnce(listResponse([story(1), story(2)], 1, 3))
      .mockResolvedValueOnce(listResponse([story(3)], 2, 3))
      .mockResolvedValueOnce(listResponse([story(4)], 3, 3));

    const stories = await adapter.listStories();

    expect(stories.map((s) => s.id)).toEqual(['story-1', 'story-2', 'story-3', 'story-4']);
    expect(fetchMock.mock.calls.map(([u]) => new URL(u as string).searchParams.get('pagination[page]'))).toEqual(['1', '2', '3']);
  });

  it('returns an empty list when no story is playable', async () => {
    fetchMock.mockResolvedValue(listResponse([], 1, 0));

    await expect(adapter.listStories()).resolves.toEqual([]);
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it('skips a malformed story instead of hiding the whole menu', async () => {
    fetchMock.mockResolvedValue(listResponse([story(1), story(2, { title: '' }), story(3, { year: 1850 }), story(4)]));

    await expect(adapter.listStories()).resolves.toMatchObject([{ id: 'story-1' }, { id: 'story-4' }]);
  });

  it('treats an invalid list envelope as broken content', async () => {
    fetchMock.mockResolvedValue(new Response(JSON.stringify({ data: { id: 1 } }), { status: 200 }));

    await expect(adapter.listStories()).rejects.toBeInstanceOf(InvalidStoryContentError);
  });

  it('reports a rejected token as an ExternalServiceError', async () => {
    fetchMock.mockResolvedValue(new Response(JSON.stringify({ error: { status: 403, message: 'Policy Failed' } }), { status: 403 }));

    await expect(adapter.listStories()).rejects.toMatchObject({ service: 'cms', status: 403 });
  });

  it('reports a 404 on the list endpoint as a CMS error, not a missing story', async () => {
    fetchMock.mockResolvedValue(new Response(JSON.stringify({ error: { status: 404, message: 'Not Found' } }), { status: 404 }));

    await expect(adapter.listStories()).rejects.toBeInstanceOf(ExternalServiceError);
  });
});
