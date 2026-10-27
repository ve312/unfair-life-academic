import { Inject, Injectable } from '@nestjs/common';
import type { StorySummary } from '../domain/entities/story-summary.entity.js';
import type { ListStoriesUseCase } from '../ports/in/list-stories.use-case.js';
import { STORY_CONTENT_PORT, type StoryContentPort } from '../ports/out/story-content.port.js';

@Injectable()
export class ListStoriesService implements ListStoriesUseCase {
  constructor(@Inject(STORY_CONTENT_PORT) private readonly storyContent: StoryContentPort) {}

  async execute(): Promise<StorySummary[]> {
    return this.storyContent.listStories();
  }
}
