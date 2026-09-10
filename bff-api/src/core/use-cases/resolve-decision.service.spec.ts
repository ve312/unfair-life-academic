import { Logger } from '@nestjs/common';
import { Test } from '@nestjs/testing';
import type { Mock } from 'vitest';
import { Character } from '../domain/entities/character.entity.js';
import { Decision } from '../domain/entities/decision.entity.js';
import { HiddenModifiers } from '../domain/entities/hidden-modifiers.entity.js';
import { MobilityAssessment, type MobilityZone } from '../domain/entities/mobility-assessment.entity.js';
import { StoryNode } from '../domain/entities/story-node.entity.js';
import {
  BaseProfileRequiredError,
  DecisionNotFoundError,
  StoryNodeNotFoundError,
} from '../domain/errors/domain.error.js';
import { toModelBaseYearDollars } from '../domain/services/consumer-price-index.js';
import { AI_PREDICTOR_PORT, type AiPredictorPort } from '../ports/out/ai-predictor.port.js';
import { RANDOM_SOURCE, type RandomSource } from '../ports/out/random-source.port.js';
import { STORY_CONTENT_PORT } from '../ports/out/story-content.port.js';
import {
  createStoryContent,
  factory,
  InMemoryStoryContent,
  intro,
  university,
} from './in-memory-story-content.fixture.js';
import { DEFAULT_STORY_YEAR, ResolveDecisionService } from './resolve-decision.service.js';

type AssessMobility = (character: Character) => Promise<MobilityAssessment>;

const BAND = { lowerBound: 0.23, upperBound: 0.79 };
const assessment = (probability: number): MobilityAssessment =>
  new MobilityAssessment({ probability, zone: MobilityAssessment.zoneOf(probability, BAND.lowerBound, BAND.upperBound) as MobilityZone, ...BAND });

/** A RandomSource that returns a fixed sequence, so every roll in a test is known. */
class SequenceRandom implements RandomSource {
  readonly calls: number[] = [];
  constructor(private readonly values: number[]) {}
  next(): number {
    const value = this.values[this.calls.length % this.values.length];
    this.calls.push(value);
    return value;
  }
}

describe('ResolveDecisionService', () => {
  let storyContent: InMemoryStoryContent;
  let assessMobility: Mock<AssessMobility>;
  let random: SequenceRandom;
  let service: ResolveDecisionService;

  const build = (rolls: number[] = [0.5]) => {
    random = new SequenceRandom(rolls);
    service = new ResolveDecisionService(storyContent, { assessMobility } satisfies AiPredictorPort, random);
  };

  beforeEach(() => {
    storyContent = createStoryContent();
    assessMobility = vi.fn<AssessMobility>();
    build();
  });

  it('follows a plain decision without calling the AI or rolling luck', async () => {
    const next = await service.execute('intro', 'work');

    expect(next).toBe(factory);
    expect(assessMobility).not.toHaveBeenCalled();
    expect(random.calls).toEqual([]);
    expect(storyContent.requestedNodeIds).toEqual(['intro', 'factory']);
  });

  it('needs no client input beyond the choice for a plain decision', async () => {
    await expect(service.execute('intro', 'work')).resolves.toBe(factory);
  });

  describe('selective classification', () => {
    it('confident mobility (p >= upper bound) -> next scene, no luck roll', async () => {
      assessMobility.mockResolvedValue(assessment(0.95));

      await expect(service.execute('intro', 'apply')).resolves.toBe(university);
      expect(random.calls).toEqual([]);
    });

    it('confident no mobility (p <= lower bound) -> failure scene, no luck roll', async () => {
      assessMobility.mockResolvedValue(assessment(0.1));

      await expect(service.execute('intro', 'apply')).resolves.toBe(factory);
      expect(random.calls).toEqual([]);
    });

    it('uncertain: a roll below the probability grants mobility', async () => {
      build([0.59]);
      assessMobility.mockResolvedValue(assessment(0.6));

      await expect(service.execute('intro', 'apply')).resolves.toBe(university);
      expect(random.calls).toEqual([0.59]);
    });

    it('uncertain: a roll at or above the probability denies mobility', async () => {
      build([0.6]);
      assessMobility.mockResolvedValue(assessment(0.6));

      await expect(service.execute('intro', 'apply')).resolves.toBe(factory);
    });

    it('uncertain: success frequency matches the probability over many rolls', async () => {
      const rolls = Array.from({ length: 1000 }, (_, i) => (i + 0.5) / 1000);
      build(rolls);
      assessMobility.mockResolvedValue(assessment(0.4));

      let successes = 0;
      for (let i = 0; i < rolls.length; i++) {
        if ((await service.execute('intro', 'apply')) === university) successes++;
      }
      expect(successes).toBe(400);
    });
  });

  describe('inflation adjustment', () => {
    const investor = new Character({
      age: 45, workclass: 'Private', education: 'Bachelors', educationNum: 13, maritalStatus: 'Married-civ-spouse',
      occupation: 'Exec-managerial', relationship: 'Husband', race: 'White', sex: 'Male', hoursPerWeek: 50,
      nativeCountry: 'Mexico', capitalGain: 20_000, capitalLoss: 1_000,
    });

    const sceneIn = (year: number | null) =>
      new StoryNode({
        id: 'intro', title: 'Intro', timerDuration: 0, isCinematic: false, year, baseProfile: investor,
        decisions: [new Decision({ id: 'apply', text: 'Apply', requiresMobilityCheck: true, nextSceneId: 'university', failureSceneId: 'factory' })],
      });

    it("deflates capital gains and losses from the scene's year to 1994 dollars before asking the AI", async () => {
      storyContent = new InMemoryStoryContent([sceneIn(2024), university, factory], {});
      build();
      assessMobility.mockResolvedValue(assessment(0.95));

      await service.execute('intro', 'apply');

      const sent = assessMobility.mock.calls[0]?.[0];
      expect(sent?.capitalGain).toBe(Math.round(toModelBaseYearDollars(20_000, 2024)));
      expect(sent?.capitalGain).toBe(9449);
      expect(sent?.capitalLoss).toBe(472);
      expect(sent?.age).toBe(45);
      expect(investor.capitalGain).toBe(20_000);
    });

    it('uses the default story year when the content has none', async () => {
      storyContent = new InMemoryStoryContent([sceneIn(null), university, factory], {});
      build();
      assessMobility.mockResolvedValue(assessment(0.95));

      await service.execute('intro', 'apply');

      expect(assessMobility.mock.calls[0]?.[0].capitalGain).toBe(Math.round(toModelBaseYearDollars(20_000, DEFAULT_STORY_YEAR)));
    });

    it('leaves 1994 amounts unchanged', async () => {
      storyContent = new InMemoryStoryContent([sceneIn(1994), university, factory], {});
      build();
      assessMobility.mockResolvedValue(assessment(0.95));

      await service.execute('intro', 'apply');

      expect(assessMobility.mock.calls[0]?.[0].capitalGain).toBe(20_000);
    });
  });

  describe('dynamic context injection (hidden modifiers)', () => {
    const borrower = new Character({
      age: 30, workclass: 'Self-emp-not-inc', education: 'HS-grad', educationNum: 9, maritalStatus: 'Never-married',
      occupation: 'Craft-repair', relationship: 'Not-in-family', race: 'White', sex: 'Female', hoursPerWeek: 40,
      nativeCountry: 'Mexico', capitalGain: 1_000,
    });

    const sceneWith = (year: number, hiddenModifiers: unknown) =>
      new StoryNode({
        id: 'bank', title: 'The bank', timerDuration: 0, isCinematic: false, year, baseProfile: borrower,
        decisions: [new Decision({
          id: 'loan', text: 'Take the loan', requiresMobilityCheck: true, nextSceneId: 'university', failureSceneId: 'factory',
          hiddenModifiers: HiddenModifiers.parse(hiddenModifiers, 'test'),
        })],
      });

    const useScene = (node: StoryNode) => {
      storyContent = new InMemoryStoryContent([node, university, factory], {});
      build();
      assessMobility.mockResolvedValue(assessment(0.95));
    };

    it('sends the mutated profile to the AI and leaves the base profile untouched', async () => {
      useScene(sceneWith(1994, { 'capital-gain': 5000, age: 2, workclass: 'Private' }));

      await service.execute('bank', 'loan');

      expect(assessMobility.mock.calls[0]?.[0]).toMatchObject({ capitalGain: 6000, age: 32, workclass: 'Private', education: 'HS-grad' });
      expect(borrower).toMatchObject({ capitalGain: 1000, age: 30, workclass: 'Self-emp-not-inc' });
    });

    it('applies the modifiers in story-year dollars, then deflates the mutated amounts to 1994', async () => {
      useScene(sceneWith(2024, { 'capital-gain': 5000 }));

      await service.execute('bank', 'loan');

      expect(assessMobility.mock.calls[0]?.[0].capitalGain).toBe(Math.round(toModelBaseYearDollars(6000, 2024)));
    });

    it('falls back to the base profile when the decision has no modifiers', async () => {
      useScene(sceneWith(1994, null));

      await service.execute('bank', 'loan');

      expect(assessMobility.mock.calls[0]?.[0]).toBe(borrower);
    });

    it('audits the applied modifiers and the exact profile sent to the AI', async () => {
      const log = vi.spyOn(Logger.prototype, 'log').mockImplementation(() => undefined);
      useScene(sceneWith(1994, { 'capital-gain': 5000, age: 80 }));

      await service.execute('bank', 'loan');

      const line = String(log.mock.calls.at(-1)?.[0]);
      const modifiers = JSON.parse(/modifiers=(\[.*?\]) modelInput=/.exec(line)?.[1] ?? 'null') as unknown;
      const modelInput = JSON.parse(/modelInput=(\{.*\})$/.exec(line)?.[1] ?? 'null') as Record<string, unknown>;
      expect(line).toContain('node=bank decision=loan year=1994');
      expect(modifiers).toEqual([
        { trait: 'capitalGain', operation: 'add', value: 5000, before: 1000, after: 6000, clamped: false },
        { trait: 'age', operation: 'add', value: 80, before: 30, after: 90, clamped: true },
      ]);
      expect(modelInput).toMatchObject({ capitalGain: 6000, age: 90, workclass: 'Self-emp-not-inc' });
      log.mockRestore();
    });

    it('audits "modifiers=none" when the decision has none', async () => {
      const log = vi.spyOn(Logger.prototype, 'log').mockImplementation(() => undefined);
      assessMobility.mockResolvedValue(assessment(0.95));

      await service.execute('intro', 'apply');

      expect(String(log.mock.calls.at(-1)?.[0])).toMatch(/modifiers=none modelInput=\{"age":19,/);
      log.mockRestore();
    });
  });

  describe('story base profile', () => {
    const farmhand = new Character({
      age: 19, workclass: 'Private', education: 'HS-grad', educationNum: 9, maritalStatus: 'Never-married',
      occupation: 'Farming-fishing', relationship: 'Own-child', race: 'Amer-Indian-Eskimo', sex: 'Male', hoursPerWeek: 50,
      nativeCountry: 'Guatemala',
    });

    const sceneOfStory = (baseProfile: Character | null, hiddenModifiers: unknown = null) =>
      new StoryNode({
        id: 'field', title: 'The field', timerDuration: 0, isCinematic: false, year: 1994, baseProfile,
        decisions: [
          new Decision({
            id: 'migrate', text: 'Migrate north', requiresMobilityCheck: true, nextSceneId: 'university', failureSceneId: 'factory',
            hiddenModifiers: HiddenModifiers.parse(hiddenModifiers, 'test'),
          }),
          new Decision({ id: 'stay', text: 'Stay home', requiresMobilityCheck: false, nextSceneId: 'factory' }),
        ],
      });

    const useScene = (node: StoryNode) => {
      storyContent = new InMemoryStoryContent([node, university, factory], {});
      build();
      assessMobility.mockResolvedValue(assessment(0.1));
    };

    it("evaluates the story's base profile; the client never supplies the character", async () => {
      useScene(sceneOfStory(farmhand));

      await expect(service.execute('field', 'migrate')).resolves.toBe(factory);
      expect(assessMobility.mock.calls[0]?.[0]).toBe(farmhand);
    });

    it('applies the hidden modifiers on top of the story base profile', async () => {
      useScene(sceneOfStory(farmhand, { 'capital-gain': 3000, workclass: 'Self-emp-not-inc' }));

      await service.execute('field', 'migrate');

      expect(assessMobility.mock.calls[0]?.[0]).toMatchObject({ nativeCountry: 'Guatemala', capitalGain: 3000, workclass: 'Self-emp-not-inc' });
      expect(farmhand.capitalGain).toBe(0);
    });

    it('rejects a mobility check in a story without base_profile with a clear error, before calling the AI', async () => {
      useScene(sceneOfStory(null));

      const error = await service.execute('field', 'migrate').catch((caught: unknown) => caught);

      expect(error).toBeInstanceOf(BaseProfileRequiredError);
      expect(error).toMatchObject({ decisionId: 'migrate', nodeId: 'field' });
      expect((error as Error).message).toContain("requires the AI, but its story has no base_profile defined");
      expect(assessMobility).not.toHaveBeenCalled();
    });

    it('follows a plain decision normally in a story without base_profile (purely narrative stories)', async () => {
      useScene(sceneOfStory(null));

      await expect(service.execute('field', 'stay')).resolves.toBe(factory);
      expect(assessMobility).not.toHaveBeenCalled();
    });
  });

  it('rejects a decision that is not offered in the current node', async () => {
    const error = await service.execute('intro', 'fly-to-the-moon').catch((caught: unknown) => caught);

    expect(error).toBeInstanceOf(DecisionNotFoundError);
    expect(error).toMatchObject({ nodeId: 'intro', decisionId: 'fly-to-the-moon' });
    expect(assessMobility).not.toHaveBeenCalled();
  });

  it('rejects any decision from an ending node', async () => {
    await expect(service.execute('factory', 'work')).rejects.toBeInstanceOf(DecisionNotFoundError);
  });

  it('propagates StoryNodeNotFoundError for an unknown current node', async () => {
    await expect(service.execute('nowhere', 'work')).rejects.toBeInstanceOf(StoryNodeNotFoundError);
  });

  it('propagates AI failures instead of guessing an outcome', async () => {
    assessMobility.mockRejectedValue(new Error('AI service unavailable'));

    await expect(service.execute('intro', 'apply')).rejects.toThrow('AI service unavailable');
    expect(storyContent.requestedNodeIds).toEqual(['intro']);
  });

  it('is resolvable through Nest dependency injection with all port tokens', async () => {
    assessMobility.mockResolvedValue(assessment(0.95));
    const moduleRef = await Test.createTestingModule({
      providers: [
        ResolveDecisionService,
        { provide: STORY_CONTENT_PORT, useValue: createStoryContent() },
        { provide: AI_PREDICTOR_PORT, useValue: { assessMobility } },
        { provide: RANDOM_SOURCE, useValue: new SequenceRandom([0.5]) },
      ],
    }).compile();

    const next = await moduleRef.get(ResolveDecisionService).execute('intro', 'apply');

    expect(next.id).toBe('university');
    expect(intro.year).toBeNull();
  });
});
