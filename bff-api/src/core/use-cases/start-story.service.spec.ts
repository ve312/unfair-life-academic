import { Test } from '@nestjs/testing';
import { InvalidStoryContentError, StoryNotFoundError } from '../domain/errors/domain.error.js';
import { STORY_CONTENT_PORT } from '../ports/out/story-content.port.js';
import { createStoryContent, intro } from './in-memory-story-content.fixture.js';
import { StartStoryService } from './start-story.service.js';

describe('StartStoryService', () => {
  it("returns the story's starting node", async () => {
    const service = new StartStoryService(createStoryContent());

    const node = await service.execute('guatemala');

    expect(node).toBe(intro);
    expect(node.audio?.url).toBe('/uploads/rain.mp3');
  });

  it('propagates StoryNotFoundError for an unknown story', async () => {
    const service = new StartStoryService(createStoryContent());

    await expect(service.execute('atlantis')).rejects.toBeInstanceOf(StoryNotFoundError);
  });

  it('propagates InvalidStoryContentError for a story without a starting scene', async () => {
    const service = new StartStoryService(createStoryContent());

    await expect(service.execute('unfinished')).rejects.toBeInstanceOf(InvalidStoryContentError);
  });

  it('is resolvable through Nest dependency injection with the port token', async () => {
    const moduleRef = await Test.createTestingModule({
      providers: [StartStoryService, { provide: STORY_CONTENT_PORT, useValue: createStoryContent() }],
    }).compile();

    const node = await moduleRef.get(StartStoryService).execute('guatemala');

    expect(node.id).toBe('intro');
  });
});
