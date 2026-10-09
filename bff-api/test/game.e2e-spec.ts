import type { INestApplication } from '@nestjs/common';
import { Test } from '@nestjs/testing';
import request from 'supertest';
import type { App } from 'supertest/types.js';
import type { Mock } from 'vitest';
import { AppModule } from '../src/app.module.js';
import type { Character } from '../src/core/domain/entities/character.entity.js';
import { MobilityAssessment, type MobilityZone } from '../src/core/domain/entities/mobility-assessment.entity.js';
import { AI_PREDICTOR_PORT } from '../src/core/ports/out/ai-predictor.port.js';
import { RANDOM_SOURCE } from '../src/core/ports/out/random-source.port.js';
import { STORY_CONTENT_PORT } from '../src/core/ports/out/story-content.port.js';
import { createStoryContent } from '../src/core/use-cases/in-memory-story-content.fixture.js';
import { ExternalServiceError } from '../src/infrastructure/adapters/external-service.error.js';
import { createValidationPipe } from '../src/infrastructure/web/validation-pipe.js';

const character = {
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
};

describe('GameController (e2e)', () => {
  let app: INestApplication<App>;
  let predictMobility: Mock<(character: Character) => Promise<MobilityAssessment>>;
  let nextRoll: number;
  const assessed = (probability: number) =>
    new MobilityAssessment({ probability, zone: MobilityAssessment.zoneOf(probability, 0.23, 0.79) as MobilityZone, lowerBound: 0.23, upperBound: 0.79 });

  beforeEach(async () => {
    predictMobility = vi.fn();
    nextRoll = 0.5;
    const moduleRef = await Test.createTestingModule({ imports: [AppModule] })
      .overrideProvider(STORY_CONTENT_PORT)
      .useValue(createStoryContent())
      .overrideProvider(AI_PREDICTOR_PORT)
      .useValue({ assessMobility: predictMobility })
      .overrideProvider(RANDOM_SOURCE)
      .useValue({ next: () => nextRoll })
      .compile();

    app = moduleRef.createNestApplication({ logger: false });
    app.useGlobalPipes(createValidationPipe());
    await app.init();
  });

  afterEach(async () => {
    await app.close();
  });

  describe('GET /health', () => {
    it('answers without calling Strapi or the AI service', async () => {
      const content = app.get(STORY_CONTENT_PORT);
      const listStories = vi.spyOn(content, 'listStories');

      await request(app.getHttpServer()).get('/health').expect(200, { status: 'ok' });
      expect(listStories).not.toHaveBeenCalled();
      expect(predictMobility).not.toHaveBeenCalled();
    });
  });

  describe('GET /game/stories', () => {
    it('lists the playable stories for the menu', async () => {
      const response = await request(app.getHttpServer()).get('/game/stories').expect(200);

      expect(response.body).toEqual([
        {
          id: 'guatemala',
          title: 'Born in Guatemala',
          shortDescription: 'A coffee-farming village, 2026.',
          coverImageUrl: '/uploads/village.jpg',
          year: 2026,
        },
        { id: 'mexico', title: 'Mexico City', shortDescription: null, coverImageUrl: null, year: 1994 },
      ]);
    });

    it('returns 503 without internal details when the CMS is unreachable', async () => {
      const content = app.get(STORY_CONTENT_PORT);
      vi.spyOn(content, 'listStories').mockRejectedValueOnce(
        new ExternalServiceError('cms', 'GET http://cms.internal:1337/api/stories failed: connect ECONNREFUSED'),
      );

      const response = await request(app.getHttpServer()).get('/game/stories').expect(503);

      expect(response.body.message).toBe('A game service is temporarily unavailable. Please try again.');
    });
  });

  describe('POST /game/start/:storyId', () => {
    it('returns the starting scene without revealing decision targets', async () => {
      const response = await request(app.getHttpServer()).post('/game/start/guatemala').expect(200);

      expect(response.body).toEqual({
        id: 'intro',
        title: 'Born in Guatemala',
        context: null,
        backgroundMediaUrl: '/uploads/village.jpg',
        mediaStartTime: null,
        mediaEndTime: null,
        audio: { url: '/uploads/rain.mp3', loop: true, defaultVolume: 0.6, startTime: null, endTime: null },
        timerDuration: 10,
        isCinematic: false,
        isEnding: false,
        economy: { year: 2026, mobilityIncomeThreshold: 111_191, cpiPublished: false },
        decisions: [
          { id: 'apply', text: 'Apply to university', requiresMobilityCheck: true },
          { id: 'work', text: 'Start working', requiresMobilityCheck: false },
        ],
      });
    });

    it('returns 404 for an unknown story', async () => {
      const response = await request(app.getHttpServer()).post('/game/start/atlantis').expect(404);

      expect(response.body).toEqual({ statusCode: 404, error: 'Not Found', message: "Story 'atlantis' was not found" });
    });

    it('returns 502 without internal details for a story with broken content', async () => {
      const response = await request(app.getHttpServer()).post('/game/start/unfinished').expect(502);

      expect(response.body.message).toBe('A game service returned an invalid response.');
    });
  });

  describe('POST /game/decision', () => {
    it('follows a plain decision without a character', async () => {
      const response = await request(app.getHttpServer())
        .post('/game/decision')
        .send({ currentNodeId: 'intro', decisionId: 'work' })
        .expect(200);

      expect(response.body).toMatchObject({ id: 'factory', isEnding: true, decisions: [] });
      expect(predictMobility).not.toHaveBeenCalled();
    });

    it('forks on the mobility check: success -> next scene', async () => {
      predictMobility.mockResolvedValue(assessed(0.95));

      const response = await request(app.getHttpServer())
        .post('/game/decision')
        .send({ currentNodeId: 'intro', decisionId: 'apply', character })
        .expect(200);

      expect(response.body.id).toBe('university');
      expect(predictMobility.mock.calls[0]?.[0]).toMatchObject({ nativeCountry: 'Guatemala', capitalGain: 0, fnlwgt: null });
    });

    it('forks on the mobility check: failure -> failure scene', async () => {
      predictMobility.mockResolvedValue(assessed(0.05));

      const response = await request(app.getHttpServer())
        .post('/game/decision')
        .send({ currentNodeId: 'intro', decisionId: 'apply', character })
        .expect(200);

      expect(response.body.id).toBe('factory');
    });

    it.each([
      ['a roll below the probability grants mobility', 0.59, 'university'],
      ['a roll above the probability denies it', 0.61, 'factory'],
    ])('uncertain band: %s', async (_label, roll, expected) => {
      predictMobility.mockResolvedValue(assessed(0.6));
      nextRoll = roll;

      const response = await request(app.getHttpServer())
        .post('/game/decision')
        .send({ currentNodeId: 'intro', decisionId: 'apply', character })
        .expect(200);

      expect(response.body.id).toBe(expected);
    });

    it("evaluates the story's base profile and ignores a character sent by an older client", async () => {
      predictMobility.mockResolvedValue(assessed(0.95));

      await request(app.getHttpServer())
        .post('/game/decision')
        .send({ currentNodeId: 'intro', decisionId: 'apply', character: { ...character, nativeCountry: 'United-States', capitalGain: 99_999 } })
        .expect(200);

      expect(predictMobility.mock.calls[0]?.[0]).toMatchObject({ nativeCountry: 'Guatemala', capitalGain: 0 });
    });

    it('returns 422 with a clear message when the story of a mobility check has no base_profile', async () => {
      const response = await request(app.getHttpServer())
        .post('/game/decision')
        .send({ currentNodeId: 'unprofiled', decisionId: 'gamble' })
        .expect(422);

      expect(response.body.message).toContain('requires the AI, but its story has no base_profile defined');
      expect(predictMobility).not.toHaveBeenCalled();
    });

    it('follows a plain decision in a story without base_profile', async () => {
      const response = await request(app.getHttpServer())
        .post('/game/decision')
        .send({ currentNodeId: 'unprofiled', decisionId: 'walk' })
        .expect(200);

      expect(response.body.id).toBe('factory');
    });

    it('returns 400 for a decision not offered in the current scene', async () => {
      const response = await request(app.getHttpServer())
        .post('/game/decision')
        .send({ currentNodeId: 'intro', decisionId: 'fly-to-the-moon' })
        .expect(400);

      expect(response.body.message).toBe("Decision 'fly-to-the-moon' is not available in story node 'intro'");
    });

    it('returns 404 for an unknown current scene', async () => {
      await request(app.getHttpServer()).post('/game/decision').send({ currentNodeId: 'nowhere', decisionId: 'work' }).expect(404);
    });

    it('rejects invalid bodies with 400 and field-level messages', async () => {
      const response = await request(app.getHttpServer())
        .post('/game/decision')
        .send({
          currentNodeId: '',
          decisionId: 'apply',
          character: { ...character, age: 12, educationNum: '9', nativeCountry: '' },
          cheat: true,
        })
        .expect(400);

      const messages = (response.body.message as string[]).join(' | ');
      expect(messages).toContain('property cheat should not exist');
      expect(messages).toContain('currentNodeId should not be empty');
      expect(messages).toContain('character.age must not be less than 17');
      expect(messages).toContain('character.educationNum must be an integer number');
      expect(messages).toContain('character.nativeCountry should not be empty');
      expect(predictMobility).not.toHaveBeenCalled();
    });

    it('rejects a character with missing traits', async () => {
      const { sex: _omitted, ...incomplete } = character;

      const response = await request(app.getHttpServer())
        .post('/game/decision')
        .send({ currentNodeId: 'intro', decisionId: 'apply', character: incomplete })
        .expect(400);

      expect((response.body.message as string[]).join(' | ')).toContain('character.sex must be a string');
    });

    it('returns 503 without internal details when the AI service is down', async () => {
      predictMobility.mockRejectedValue(
        new ExternalServiceError('ai-service', 'POST http://ai.internal:8000/predict failed: connect ECONNREFUSED'),
      );

      const response = await request(app.getHttpServer())
        .post('/game/decision')
        .send({ currentNodeId: 'intro', decisionId: 'apply', character })
        .expect(503);

      expect(response.body).toEqual({
        statusCode: 503,
        error: 'Service Unavailable',
        message: 'A game service is temporarily unavailable. Please try again.',
      });
    });
  });
});
