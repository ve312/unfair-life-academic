import { Character } from '../entities/character.entity.js';
import { Decision } from '../entities/decision.entity.js';
import { InvalidMobilityAssessmentError, MobilityAssessment } from '../entities/mobility-assessment.entity.js';
import { StoryNode } from '../entities/story-node.entity.js';
import { InvalidStoryContentError } from '../errors/domain.error.js';
import {
  consumerPriceIndex,
  isPublishedYear,
  LAST_PUBLISHED_YEAR,
  mobilityIncomeThreshold,
  MODEL_BASE_YEAR,
  PROJECTION_RATE,
  toModelBaseYearDollars,
} from './consumer-price-index.js';
import { deflateToModelBaseYear } from './inflation-adjuster.js';
import { InvalidRandomValueError, resolveMobility } from './systemic-luck.js';

describe('consumer price index', () => {
  it('uses published BLS CPI-U annual averages', () => {
    expect(consumerPriceIndex(1994)).toBe(148.2);
    expect(consumerPriceIndex(2024)).toBe(313.689);
    expect(isPublishedYear(2024)).toBe(true);
    expect(LAST_PUBLISHED_YEAR).toBe(2024);
  });

  it(`projects years after ${LAST_PUBLISHED_YEAR} at ${PROJECTION_RATE * 100}% per year`, () => {
    expect(isPublishedYear(2026)).toBe(false);
    expect(consumerPriceIndex(2025)).toBeCloseTo(313.689 * 1.025, 10);
    expect(consumerPriceIndex(2026)).toBeCloseTo(313.689 * 1.025 ** 2, 10);
  });

  it('converts story-year dollars to 1994 dollars', () => {
    expect(toModelBaseYearDollars(10_000, MODEL_BASE_YEAR)).toBe(10_000);
    expect(toModelBaseYearDollars(313_689, 2024)).toBeCloseTo(148_200, 6);
  });

  it("expresses the model's $50K (1994) in story-year dollars", () => {
    expect(mobilityIncomeThreshold(1994)).toBe(50_000);
    expect(mobilityIncomeThreshold(2024)).toBe(105_833);
    expect(mobilityIncomeThreshold(2026)).toBe(111_191);
  });

  it.each([1989, 2101, 2024.5, Number.NaN])('rejects unsupported year %s', (year) => {
    expect(() => consumerPriceIndex(year)).toThrow(InvalidStoryContentError);
  });
});

describe('deflateToModelBaseYear', () => {
  const character = new Character({
    age: 40, workclass: 'Private', education: 'Masters', educationNum: 14, maritalStatus: 'Married-civ-spouse',
    occupation: 'Prof-specialty', relationship: 'Wife', race: 'White', sex: 'Female', hoursPerWeek: 45,
    nativeCountry: 'Peru', capitalGain: 15_000, capitalLoss: 2_000,
  });

  it('deflates only capital gains and losses, rounding to whole dollars', () => {
    const deflated = deflateToModelBaseYear(character, 2024);

    expect(deflated.capitalGain).toBe(7087);
    expect(deflated.capitalLoss).toBe(945);
    const unchanged = ['age', 'workclass', 'education', 'educationNum', 'maritalStatus', 'occupation', 'relationship',
      'race', 'sex', 'hoursPerWeek', 'nativeCountry', 'fnlwgt'] as const;
    unchanged.forEach((trait) => expect(deflated[trait]).toBe(character[trait]));
  });

  it('returns the same character for 1994', () => {
    expect(deflateToModelBaseYear(character, 1994)).toBe(character);
  });
});

describe('MobilityAssessment', () => {
  const band = { lowerBound: 0.23, upperBound: 0.79 };

  it.each([
    [0.79, 'mobility'],
    [0.99, 'mobility'],
    [0.23, 'no_mobility'],
    [0.0, 'no_mobility'],
    [0.5, 'uncertain'],
  ] as const)('places probability %s in zone %s', (probability, zone) => {
    expect(MobilityAssessment.zoneOf(probability, band.lowerBound, band.upperBound)).toBe(zone);
    expect(new MobilityAssessment({ probability, zone, ...band }).zone).toBe(zone);
  });

  it.each([
    ['a zone that contradicts the probability', { probability: 0.5, zone: 'mobility' }],
    ['a probability above 1', { probability: 1.2, zone: 'mobility' }],
    ['an inverted band', { probability: 0.5, zone: 'uncertain', lowerBound: 0.8, upperBound: 0.2 }],
    ['an unknown zone', { probability: 0.5, zone: 'maybe' }],
  ])('rejects %s', (_label, override) => {
    expect(() => new MobilityAssessment({ ...band, ...override } as never)).toThrow(InvalidMobilityAssessmentError);
  });
});

describe('resolveMobility (systemic luck)', () => {
  const band = { lowerBound: 0.23, upperBound: 0.79 };
  const fixed = (value: number) => ({ next: () => value });
  const never = { next: (): number => { throw new Error('the model was confident; luck must not be rolled'); } };

  it('lets the model decide the extremes without rolling', () => {
    const yes = resolveMobility(new MobilityAssessment({ probability: 0.9, zone: 'mobility', ...band }), never);
    const no = resolveMobility(new MobilityAssessment({ probability: 0.1, zone: 'no_mobility', ...band }), never);

    expect(yes).toMatchObject({ hasMobility: true, decidedBy: 'model', roll: null });
    expect(no).toMatchObject({ hasMobility: false, decidedBy: 'model', roll: null });
  });

  it('flips a coin weighted by the probability inside the band', () => {
    const uncertain = new MobilityAssessment({ probability: 0.6, zone: 'uncertain', ...band });

    expect(resolveMobility(uncertain, fixed(0.599))).toMatchObject({ hasMobility: true, decidedBy: 'luck', roll: 0.599 });
    expect(resolveMobility(uncertain, fixed(0.6))).toMatchObject({ hasMobility: false, decidedBy: 'luck', roll: 0.6 });
  });

  it.each([1, -0.1, Number.NaN])('rejects an invalid random value %s', (value) => {
    const uncertain = new MobilityAssessment({ probability: 0.6, zone: 'uncertain', ...band });

    expect(() => resolveMobility(uncertain, fixed(value))).toThrow(InvalidRandomValueError);
  });
});

describe('StoryNode year', () => {
  const decision = new Decision({ id: 'd', text: 'Go', requiresMobilityCheck: false, nextSceneId: 'n' });
  const node = (year: unknown) => new StoryNode({ id: 's', title: 'S', timerDuration: 0, isCinematic: false, decisions: [decision], year: year as number });

  it('accepts a supported year or none', () => {
    expect(node(2026).year).toBe(2026);
    expect(node(null).year).toBeNull();
    expect(node(undefined).year).toBeNull();
  });

  it.each([1989, 2101, 2026.5, '2026'])('rejects year %s', (year) => {
    expect(() => node(year)).toThrow(InvalidStoryContentError);
  });
});
