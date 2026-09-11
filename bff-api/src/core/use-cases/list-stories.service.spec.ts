import { Test } from '@nestjs/testing';
import { STORY_CONTENT_PORT } from '../ports/out/story-content.port.js';
import { createStoryContent, InMemoryStoryContent, storySummaries } from './in-memory-story-content.fixture.js';
import { ListStoriesService } from './list-stories.service.js';

describe('ListStoriesService', () => {
  it('returns the playable stories from the content port', async () => {
    const service = new ListStoriesService(createStoryContent());

    await expect(service.execute()).resolves.toEqual(storySummaries);
  });

  it('returns an empty list when no story is playable', async () => {
    const service = new ListStoriesService(new InMemoryStoryContent([], {}));

    await expect(service.execute()).resolves.toEqual([]);
  });

  it('propagates content-source failures', async () => {
    const content = createStoryContent();
    vi.spyOn(content, 'listStories').mockRejectedValue(new Error('CMS down'));

    await expect(new ListStoriesService(content).execute()).rejects.toThrow('CMS down');
  });

  it('is resolvable through Nest dependency injection with the port token', async () => {
    const moduleRef = await Test.createTestingModule({
      providers: [ListStoriesService, { provide: STORY_CONTENT_PORT, useValue: createStoryContent() }],
    }).compile();

    await expect(moduleRef.get(ListStoriesService).execute()).resolves.toHaveLength(2);
  });
});
