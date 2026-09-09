import { Inject, Injectable } from '@nestjs/common';
import type { StoryNode } from '../domain/entities/story-node.entity.js';
import type { StartStoryUseCase } from '../ports/in/start-story.use-case.js';
import { STORY_CONTENT_PORT, type StoryContentPort } from '../ports/out/story-content.port.js';

@Injectable()
export class StartStoryService implements StartStoryUseCase {
  constructor(@Inject(STORY_CONTENT_PORT) private readonly storyContent: StoryContentPort) {}

  async execute(storyId: string): Promise<StoryNode> {
    return this.storyContent.getStartingNode(storyId);
  }
}
