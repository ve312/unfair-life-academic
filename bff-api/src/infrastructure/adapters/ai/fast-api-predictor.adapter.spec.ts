import type { Mock } from 'vitest';
import { Character, type CharacterProps } from '../../../core/domain/entities/character.entity.js';
import { MobilityAssessment } from '../../../core/domain/entities/mobility-assessment.entity.js';
import type { AiServiceConfig } from '../../config/external-services.config.js';
import { ExternalServiceError } from '../external-service.error.js';
import { FastApiPredictorAdapter, toPredictionPayload } from './fast-api-predictor.adapter.js';

const config: AiServiceConfig = { baseUrl: 'http://ai.internal:8000', token: 'shared-secret', timeoutMs: 1000 };

const characterProps: CharacterProps = {
  age: 34,
  workclass: 'Private',
  education: 'HS-grad',
  educationNum: 9,
  maritalStatus: 'Married-civ-spouse',
  occupation: 'Adm-clerical',
  relationship: 'Wife',
  race: 'White',
  sex: 'Female',
  hoursPerWeek: 40,
  nativeCountry: 'Mexico',
};
const character = new Character(characterProps);

const jsonResponse = (status: number, body: unknown): Response =>
  new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } });

describe('toPredictionPayload', () => {
  it('maps every trait to the snake_case FastAPI schema and omits an unset fnlwgt', () => {
    expect(toPredictionPayload(character)).toEqual({
      age: 34,
      workclass: 'Private',
      education: 'HS-grad',
      education_num: 9,
      marital_status: 'Married-civ-spouse',
      occupation: 'Adm-clerical',
      relationship: 'Wife',
      race: 'White',
      sex: 'Female',
      hours_per_week: 40,
      native_country: 'Mexico',
      capital_gain: 0,
      capital_loss: 0,
    });
  });

  it('includes fnlwgt when the character has one', () => {
    const weighted = new Character({ ...characterProps, fnlwgt: 178211 });

    expect(toPredictionPayload(weighted).fnlwgt).toBe(178211);
  });
});

describe('FastApiPredictorAdapter', () => {
  let fetchMock: Mock<typeof fetch>;
  let adapter: FastApiPredictorAdapter;

  beforeEach(() => {
    fetchMock = vi.fn<typeof fetch>();
    vi.stubGlobal('fetch', fetchMock);
    adapter = new FastApiPredictorAdapter(config);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  const prediction = (probability: number, zone: string) => ({
    probability, zone, lower_bound: 0.23, upper_bound: 0.79, threshold: 0.7825, has_mobility: probability >= 0.7825,
  });

  it('POSTs the character as JSON and returns the mobility assessment', async () => {
    fetchMock.mockResolvedValue(jsonResponse(200, prediction(0.97, 'mobility')));

    await expect(adapter.assessMobility(character)).resolves.toEqual(
      new MobilityAssessment({ probability: 0.97, zone: 'mobility', lowerBound: 0.23, upperBound: 0.79 }),
    );

    const [url, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(url).toBe('http://ai.internal:8000/predict');
    expect(init.method).toBe('POST');
    expect((init.headers as Record<string, string>)['Content-Type']).toBe('application/json');
    expect((init.headers as Record<string, string>)['X-BFF-Token']).toBe('shared-secret');
    expect(JSON.parse(init.body as string)).toEqual(toPredictionPayload(character));
  });

  it.each([
    [0.05, 'no_mobility'],
    [0.45, 'uncertain'],
  ])('maps probability %s to zone %s', async (probability, zone) => {
    fetchMock.mockResolvedValue(jsonResponse(200, prediction(probability, zone)));

    await expect(adapter.assessMobility(character)).resolves.toMatchObject({ probability, zone });
  });

  it('rejects an assessment whose zone contradicts its probability', async () => {
    fetchMock.mockResolvedValue(jsonResponse(200, prediction(0.45, 'mobility')));

    await expect(adapter.assessMobility(character)).rejects.toThrow(/inconsistent assessment/);
  });

  it('reports a validation rejection with FastAPI detail', async () => {
    fetchMock.mockResolvedValue(jsonResponse(422, { detail: "'age' must be between 17 and 90, got 12" }));

    const error = await adapter.assessMobility(character).catch((caught: unknown) => caught);

    expect(error).toBeInstanceOf(ExternalServiceError);
    expect(error).toMatchObject({ service: 'ai-service', status: 422 });
    expect((error as Error).message).toContain("'age' must be between 17 and 90");
  });

  it('reports the model being unavailable (503)', async () => {
    fetchMock.mockResolvedValue(jsonResponse(503, { detail: 'Prediction model is not loaded' }));

    await expect(adapter.assessMobility(character)).rejects.toThrow('responded 503: Prediction model is not loaded');
  });

  it('rejects an unexpected response body instead of guessing', async () => {
    fetchMock.mockResolvedValue(jsonResponse(200, { has_mobility: true, probability: 0.97 }));

    await expect(adapter.assessMobility(character)).rejects.toThrow(/unexpected body/);
  });

  it('rejects a non-JSON response', async () => {
    fetchMock.mockResolvedValue(new Response('<html>Bad Gateway</html>', { status: 502 }));

    await expect(adapter.assessMobility(character)).rejects.toThrow('returned non-JSON content (status 502)');
  });

  it('wraps an unreachable AI service in ExternalServiceError', async () => {
    fetchMock.mockRejectedValue(new TypeError('fetch failed', { cause: new Error('connect ECONNREFUSED 127.0.0.1:8000') }));

    await expect(adapter.assessMobility(character)).rejects.toThrow(
      '[ai-service] POST http://ai.internal:8000/predict failed: connect ECONNREFUSED 127.0.0.1:8000',
    );
  });

  it('wraps a timeout in ExternalServiceError', async () => {
    fetchMock.mockRejectedValue(new DOMException('The operation was aborted due to timeout', 'TimeoutError'));

    await expect(adapter.assessMobility(character)).rejects.toThrow('timed out after 1000 ms');
  });
});
