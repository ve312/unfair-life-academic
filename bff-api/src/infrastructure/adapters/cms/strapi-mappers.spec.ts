import { InvalidStoryContentError } from '../../../core/domain/errors/domain.error.js';
import { mapMediaTrim, mapMediaUrl, mapStartingNode, mapStoryNode, mapStorySummary, unwrapStrapiData, unwrapStrapiList } from './strapi-mappers.js';

const MEDIA_BASE = 'http://localhost:1337';

/** Shape returned by Strapi v5 for GET /api/scenes/:documentId with the adapter's populate. */
const rawScene = () => ({
  id: 1,
  documentId: 'intro-doc',
  title: 'Intro',
  timer_duration: 10,
  is_cinematic: false,
  background_media: { id: 3, documentId: 'media-doc', url: '/uploads/village.jpg' },
  audio_track: {
    id: 1,
    documentId: 'track-doc',
    loop: true,
    default_volume: 0.6,
    audio_file: { id: 1, documentId: 'file-doc', url: '/uploads/rain.mp3' },
  },
  available_decisions: [
    {
      id: 1,
      documentId: 'apply-doc',
      text: 'Apply to university',
      requires_mobility_check: true,
      next_scene: { id: 2, documentId: 'university-doc', title: 'University' },
      failure_scene: { id: 3, documentId: 'factory-doc', title: 'Factory job' },
    },
    {
      id: 2,
      documentId: 'work-doc',
      text: 'Start working',
      requires_mobility_check: false,
      next_scene: { id: 3, documentId: 'factory-doc', title: 'Factory job' },
      failure_scene: null,
    },
  ],
});

describe('strapi-mappers', () => {
  it('maps a fully populated scene to a StoryNode', () => {
    const node = mapStoryNode(rawScene(), MEDIA_BASE);

    expect(node.id).toBe('intro-doc');
    expect(node.title).toBe('Intro');
    expect(node.timerDuration).toBe(10);
    expect(node.isCinematic).toBe(false);
    expect(node.backgroundMediaUrl).toBe('http://localhost:1337/uploads/village.jpg');
    expect(node.audio).toMatchObject({ url: 'http://localhost:1337/uploads/rain.mp3', loop: true, defaultVolume: 0.6 });
    expect(node.decisions.map((decision) => decision.id)).toEqual(['apply-doc', 'work-doc']);
    expect(node.findDecision('apply-doc')?.resolveNextSceneId(false)).toBe('factory-doc');
    expect(node.findDecision('work-doc')?.failureSceneId).toBeNull();
  });

  it('maps a scene without media, audio or decisions as a silent ending', () => {
    const node = mapStoryNode(
      { documentId: 'end', title: 'The End', timer_duration: 0, is_cinematic: true, background_media: null, audio_track: null, available_decisions: [] },
      MEDIA_BASE,
    );

    expect(node.isEnding).toBe(true);
    expect(node.backgroundMediaUrl).toBeNull();
    expect(node.audio).toBeNull();
  });

  it('keeps absolute media URLs from cloud upload providers', () => {
    expect(mapMediaUrl({ url: 'https://cdn.example.com/rain.mp3' }, MEDIA_BASE, 'audio')).toBe(
      'https://cdn.example.com/rain.mp3',
    );
  });

  it('accepts decimal volumes returned as strings (PostgreSQL)', () => {
    const scene = rawScene();
    scene.audio_track.default_volume = '0.25' as unknown as number;

    expect(mapStoryNode(scene, MEDIA_BASE).audio?.defaultVolume).toBe(0.25);
  });

  it('defaults timer_duration and is_cinematic when Strapi returns null', () => {
    const scene = { ...rawScene(), timer_duration: null, is_cinematic: null };

    const node = mapStoryNode(scene, MEDIA_BASE);

    expect(node.timerDuration).toBe(0);
    expect(node.isCinematic).toBe(false);
  });

  describe('year', () => {
    it("prefers the scene's own year (time jumps)", () => {
      expect(mapStoryNode({ ...rawScene(), year: 2046, story: { documentId: 'st', year: 2026 } }, MEDIA_BASE).year).toBe(2046);
    });

    it("falls back to the scene's story year", () => {
      expect(mapStoryNode({ ...rawScene(), year: null, story: { documentId: 'st', year: 2026 } }, MEDIA_BASE).year).toBe(2026);
    });

    it('falls back to the started story when the scene is not linked to one', () => {
      const node = mapStartingNode({ documentId: 'st', year: 2030, starting_scene: { ...rawScene(), story: null } }, MEDIA_BASE);
      expect(node.year).toBe(2030);
    });

    it('is null when no year is known', () => {
      expect(mapStoryNode(rawScene(), MEDIA_BASE).year).toBeNull();
    });

    it.each([['a non-integer year', 2026.5], ['a year as text', '2026'], ['an unsupported year', 1850]])('rejects %s', (_label, year) => {
      expect(() => mapStoryNode({ ...rawScene(), year }, MEDIA_BASE)).toThrow(InvalidStoryContentError);
    });
  });

  it('maps a story to its starting scene', () => {
    const node = mapStartingNode({ documentId: 'story-doc', title: 'Born in Guatemala', starting_scene: rawScene() }, MEDIA_BASE);

    expect(node.id).toBe('intro-doc');
  });

  it.each([
    ['a decision without next_scene', (scene: ReturnType<typeof rawScene>) => {
      (scene.available_decisions[1] as Record<string, unknown>).next_scene = null;
    }],
    ['a mobility check without failure_scene', (scene: ReturnType<typeof rawScene>) => {
      (scene.available_decisions[0] as Record<string, unknown>).failure_scene = null;
    }],
    ['decisions that were not populated', (scene: ReturnType<typeof rawScene>) => {
      delete (scene as Partial<ReturnType<typeof rawScene>>).available_decisions;
    }],
    ['an audio track without a file', (scene: ReturnType<typeof rawScene>) => {
      (scene.audio_track as Record<string, unknown>).audio_file = null;
    }],
    ['a volume above 1', (scene: ReturnType<typeof rawScene>) => {
      scene.audio_track.default_volume = 1.5;
    }],
    ['a non-integer timer', (scene: ReturnType<typeof rawScene>) => {
      scene.timer_duration = 2.5;
    }],
  ])('rejects %s with InvalidStoryContentError', (_label, corrupt) => {
    const scene = rawScene();
    corrupt(scene);

    expect(() => mapStoryNode(scene, MEDIA_BASE)).toThrow(InvalidStoryContentError);
  });

  describe('base_profile', () => {
    const profile = {
      age: 19, workclass: 'Private', education: 'HS-grad', 'education-num': 9, 'marital-status': 'Never-married',
      occupation: 'Farming-fishing', relationship: 'Own-child', race: 'Amer-Indian-Eskimo', sex: 'Male', 'hours-per-week': 50,
      'native-country': 'Guatemala', 'capital-gain': 0, 'capital-loss': 0,
    };

    it("takes the base profile of the scene's story", () => {
      const node = mapStoryNode({ ...rawScene(), story: { year: 1998, base_profile: profile } }, MEDIA_BASE);

      expect(node.baseProfile).toMatchObject({ age: 19, nativeCountry: 'Guatemala', educationNum: 9 });
    });

    it('falls back to the started story when the starting scene is not linked to one', () => {
      const node = mapStartingNode({ documentId: 'story-doc', title: 'Born in Guatemala', base_profile: profile, starting_scene: rawScene() }, MEDIA_BASE);

      expect(node.baseProfile?.nativeCountry).toBe('Guatemala');
    });

    it('is null for a story created before the field existed', () => {
      expect(mapStoryNode({ ...rawScene(), story: { year: 1998, base_profile: null } }, MEDIA_BASE).baseProfile).toBeNull();
      expect(mapStoryNode(rawScene(), MEDIA_BASE).baseProfile).toBeNull();
    });

    it('names the scene when the profile is invalid', () => {
      expect(() => mapStoryNode({ ...rawScene(), story: { year: 1998, base_profile: { age: 30 } } }, MEDIA_BASE)).toThrow(
        "Scene 'intro-doc': story base_profile: missing traits",
      );
    });
  });

  describe('context', () => {
    it("maps the scene's narrative context, trimmed and with its line breaks", () => {
      const scene = { ...rawScene(), context: '  1998, Guatemala City.\nAna is 19.  ' };

      expect(mapStoryNode(scene, MEDIA_BASE).context).toBe('1998, Guatemala City.\nAna is 19.');
    });

    it.each([undefined, null, '', '   '])('treats %j as no context', (value) => {
      expect(mapStoryNode({ ...rawScene(), context: value }, MEDIA_BASE).context).toBeNull();
    });

    it('rejects a context that is not text', () => {
      expect(() => mapStoryNode({ ...rawScene(), context: 42 }, MEDIA_BASE)).toThrow("Story node 'intro-doc': 'context' must be text");
    });
  });

  describe('hidden_modifiers', () => {
    it('maps the JSON field of a mobility check to HiddenModifiers', () => {
      const scene = rawScene();
      (scene.available_decisions[0] as Record<string, unknown>).hidden_modifiers = { 'capital-gain': 5000, workclass: 'Private' };

      const [apply] = mapStoryNode(scene, MEDIA_BASE).decisions;

      expect(apply?.hiddenModifiers.deltas).toEqual([['capitalGain', 5000]]);
      expect(apply?.hiddenModifiers.replacements).toEqual([['workclass', 'Private']]);
    });

    it('treats a missing or null field as no modifiers', () => {
      const scene = rawScene();
      (scene.available_decisions[0] as Record<string, unknown>).hidden_modifiers = null;

      expect(mapStoryNode(scene, MEDIA_BASE).decisions.every((decision) => decision.hiddenModifiers.isEmpty)).toBe(true);
    });

    it('names the scene and decision when a modifier is invalid', () => {
      const scene = rawScene();
      (scene.available_decisions[0] as Record<string, unknown>).hidden_modifiers = { salary: 100 };

      expect(() => mapStoryNode(scene, MEDIA_BASE)).toThrow(
        "Scene 'intro-doc', decision 'apply-doc': hidden_modifiers: 'salary' is not a modifiable trait",
      );
    });
  });

  it('names the offending scene and decision in the error', () => {
    const scene = rawScene();
    (scene.available_decisions[1] as Record<string, unknown>).next_scene = null;

    expect(() => mapStoryNode(scene, MEDIA_BASE)).toThrow("Scene 'intro-doc', decision 'work-doc': 'next_scene' is not set");
  });

  it('rejects a story without a starting scene', () => {
    expect(() => mapStartingNode({ documentId: 'story-doc', starting_scene: null }, MEDIA_BASE)).toThrow(
      "Story 'story-doc' has no starting scene",
    );
  });

  it('unwraps the Strapi data envelope and rejects a list response', () => {
    expect(unwrapStrapiData({ data: { documentId: 'x' }, meta: {} }, 'Scene')).toEqual({ documentId: 'x' });
    expect(() => unwrapStrapiData({ data: [], meta: {} }, 'Scene')).toThrow(InvalidStoryContentError);
  });
});

describe('mapStorySummary', () => {
  const rawStory = () => ({
    id: 7,
    documentId: 'story-doc',
    title: 'Born in Guatemala',
    short_description: 'A coffee-farming village, 2026.',
    year: 2026,
    cover_image: { id: 3, documentId: 'img-doc', url: '/uploads/cover.jpg' },
  });

  it('maps a Strapi list entry to a StorySummary with an absolute cover URL', () => {
    expect(mapStorySummary(rawStory(), MEDIA_BASE)).toMatchObject({
      id: 'story-doc',
      title: 'Born in Guatemala',
      shortDescription: 'A coffee-farming village, 2026.',
      coverImageUrl: 'http://localhost:1337/uploads/cover.jpg',
      year: 2026,
    });
  });

  it('maps a story without cover, description or year', () => {
    const summary = mapStorySummary({ documentId: 'story-doc', title: 'T', short_description: null, cover_image: null, year: null }, MEDIA_BASE);

    expect(summary).toMatchObject({ shortDescription: null, coverImageUrl: null, year: null });
  });

  it.each([
    ['a missing documentId', { documentId: undefined }],
    ['a missing title', { title: null }],
    ['a cover without url', { cover_image: { id: 3 } }],
    ['a year as text', { year: '2026' }],
  ])('rejects %s', (_label, override) => {
    expect(() => mapStorySummary({ ...rawStory(), ...override }, MEDIA_BASE)).toThrow(InvalidStoryContentError);
  });
});

describe('unwrapStrapiList', () => {
  it('returns the entries and the page count', () => {
    expect(unwrapStrapiList({ data: [{ id: 1 }], meta: { pagination: { pageCount: 4 } } }, 'list')).toEqual({ items: [{ id: 1 }], pageCount: 4 });
  });

  it('treats a missing pagination block as one page', () => {
    expect(unwrapStrapiList({ data: [] }, 'list')).toEqual({ items: [], pageCount: 1 });
  });

  it('rejects a findOne-shaped body', () => {
    expect(() => unwrapStrapiList({ data: { id: 1 } }, 'list')).toThrow(InvalidStoryContentError);
  });
});

describe('media trims', () => {
  it('maps the background and audio fragments of a scene', () => {
    const node = mapStoryNode(
      { ...rawScene(), media_start_time: 12.5, media_end_time: 30, audio_start_time: 4, audio_end_time: null },
      MEDIA_BASE,
    );

    expect(node.backgroundTrim).toMatchObject({ startTime: 12.5, endTime: 30 });
    expect(node.audioTrim).toMatchObject({ startTime: 4, endTime: null });
  });

  it('plays whole files when no trim is set', () => {
    const node = mapStoryNode(rawScene(), MEDIA_BASE);

    expect(node.backgroundTrim.isFullLength).toBe(true);
    expect(node.audioTrim.isFullLength).toBe(true);
  });

  it('accepts floats returned as strings (PostgreSQL)', () => {
    expect(mapMediaTrim('1.25', '9.5', 'trim')).toMatchObject({ startTime: 1.25, endTime: 9.5 });
  });

  it.each([
    ['an end before the start', 20, 10],
    ['an end equal to the start', 10, 10],
    ['a negative start', -1, null],
    ['a non-numeric value', 'soon', null],
  ])('rejects %s, naming the scene', (_label, start, end) => {
    expect(() => mapStoryNode({ ...rawScene(), media_start_time: start, media_end_time: end }, MEDIA_BASE)).toThrow(
      /Scene 'intro-doc': media trim/,
    );
  });
});
