import { AudioConfig } from '../domain/entities/audio-config.entity.js';
import { Character } from '../domain/entities/character.entity.js';
import { Decision } from '../domain/entities/decision.entity.js';
import { StoryNode } from '../domain/entities/story-node.entity.js';
import { StorySummary } from '../domain/entities/story-summary.entity.js';
import { InvalidStoryContentError, StoryNodeNotFoundError, StoryNotFoundError } from '../domain/errors/domain.error.js';
import type { StoryContentPort } from '../ports/out/story-content.port.js';

/**
 * Test fixture: a small story graph held in memory.
 *
 *   intro --"Apply to university" (mobility check)--> university | factory (failure)
 *   intro --"Start working"-----------------------------> factory
 *   university, factory: endings
 *
 * The story's base profile (the character of every mobility check) is `player`.
 */
/** Base profile of the fixture story: a 19-year-old from Guatemala. */
export const player = new Character({
  age: 19,
  workclass: 'Private',
  education: 'HS-grad',
  educationNum: 9,
  maritalStatus: 'Never-married',
  occupation: 'Handlers-cleaners',
  relationship: 'Own-child',
  race: 'Amer-Indian-Eskimo',
  sex: 'Male',
  hoursPerWeek: 40,
  nativeCountry: 'Guatemala',
});

export const intro = new StoryNode({
  id: 'intro',
  title: 'Born in Guatemala',
  backgroundMediaUrl: '/uploads/village.jpg',
  audio: new AudioConfig({ url: '/uploads/rain.mp3', loop: true, defaultVolume: 0.6 }),
  timerDuration: 10,
  isCinematic: false,
  baseProfile: player,
  decisions: [
    new Decision({
      id: 'apply',
      text: 'Apply to university',
      requiresMobilityCheck: true,
      nextSceneId: 'university',
      failureSceneId: 'factory',
    }),
    new Decision({ id: 'work', text: 'Start working', requiresMobilityCheck: false, nextSceneId: 'factory' }),
  ],
});

export const university = new StoryNode({
  id: 'university',
  title: 'University',
  timerDuration: 0,
  isCinematic: true,
  decisions: [],
});

export const factory = new StoryNode({
  id: 'factory',
  title: 'Factory job',
  timerDuration: 0,
  isCinematic: true,
  decisions: [],
});


export class InMemoryStoryContent implements StoryContentPort {
  private readonly nodes = new Map<string, StoryNode>();
  readonly requestedNodeIds: string[] = [];

  constructor(
    nodes: StoryNode[],
    private readonly startingNodeIdByStory: Record<string, string | null>,
    private readonly summaries: StorySummary[] = [],
  ) {
    nodes.forEach((node) => this.nodes.set(node.id, node));
  }

  async getStoryNode(nodeId: string): Promise<StoryNode> {
    this.requestedNodeIds.push(nodeId);
    const node = this.nodes.get(nodeId);
    if (node === undefined) {
      throw new StoryNodeNotFoundError(nodeId);
    }
    return node;
  }

  async listStories(): Promise<StorySummary[]> {
    return [...this.summaries];
  }

  async getStartingNode(storyId: string): Promise<StoryNode> {
    if (!(storyId in this.startingNodeIdByStory)) {
      throw new StoryNotFoundError(storyId);
    }
    const startingNodeId = this.startingNodeIdByStory[storyId];
    if (startingNodeId === null) {
      throw new InvalidStoryContentError(`Story '${storyId}' has no starting scene`);
    }
    return this.getStoryNode(startingNodeId);
  }
}

export const storySummaries = [
  new StorySummary({
    id: 'guatemala',
    title: 'Born in Guatemala',
    shortDescription: 'A coffee-farming village, 2026.',
    coverImageUrl: '/uploads/village.jpg',
    year: 2026,
  }),
  new StorySummary({ id: 'mexico', title: 'Mexico City', year: 1994 }),
];

/** A scene whose story declares no base_profile but which still offers a mobility check (a content error). */
export const unprofiled = new StoryNode({
  id: 'unprofiled',
  title: 'A story without a character',
  timerDuration: 0,
  isCinematic: false,
  decisions: [
    new Decision({ id: 'gamble', text: 'Try your luck', requiresMobilityCheck: true, nextSceneId: 'university', failureSceneId: 'factory' }),
    new Decision({ id: 'walk', text: 'Keep walking', requiresMobilityCheck: false, nextSceneId: 'factory' }),
  ],
});

export const createStoryContent = (): InMemoryStoryContent =>
  new InMemoryStoryContent([intro, unprofiled, university, factory], { guatemala: 'intro', unfinished: null }, storySummaries);
