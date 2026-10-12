import { Test } from '@nestjs/testing';
import { StorySummary } from '../../../core/domain/entities/story-summary.entity.js';
import { LIST_STORIES_USE_CASE } from '../../../core/ports/in/list-stories.use-case.js';
import { RESOLVE_DECISION_USE_CASE } from '../../../core/ports/in/resolve-decision.use-case.js';
import { START_STORY_USE_CASE } from '../../../core/ports/in/start-story.use-case.js';
import { GameController } from './game.controller.js';

describe('GameController.stories', () => {
  const listStories = { execute: vi.fn<() => Promise<StorySummary[]>>() };

  const controller = async () => {
    const moduleRef = await Test.createTestingModule({
      controllers: [GameController],
      providers: [
        { provide: LIST_STORIES_USE_CASE, useValue: listStories },
        { provide: START_STORY_USE_CASE, useValue: { execute: vi.fn() } },
        { provide: RESOLVE_DECISION_USE_CASE, useValue: { execute: vi.fn() } },
      ],
    }).compile();
    return moduleRef.get(GameController);
  };

  beforeEach(() => listStories.execute.mockReset());

  it('maps each story summary to the menu DTO', async () => {
    listStories.execute.mockResolvedValue([
      new StorySummary({ id: 'doc-1', title: 'Born in Guatemala', shortDescription: 'A village.', coverImageUrl: 'http://cms/uploads/v.jpg', year: 2026 }),
      new StorySummary({ id: 'doc-2', title: 'Mexico City' }),
    ]);

    await expect((await controller()).stories()).resolves.toEqual([
      { id: 'doc-1', title: 'Born in Guatemala', shortDescription: 'A village.', coverImageUrl: 'http://cms/uploads/v.jpg', year: 2026 },
      { id: 'doc-2', title: 'Mexico City', shortDescription: null, coverImageUrl: null, year: null },
    ]);
  });

  it('returns an empty array when there are no stories', async () => {
    listStories.execute.mockResolvedValue([]);

    await expect((await controller()).stories()).resolves.toEqual([]);
  });
});
