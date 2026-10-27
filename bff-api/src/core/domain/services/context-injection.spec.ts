import { Character } from '../entities/character.entity.js';
import { HiddenModifiers } from '../entities/hidden-modifiers.entity.js';
import { InvalidStoryContentError } from '../errors/domain.error.js';
import { injectContext } from './context-injection.js';

const CONTEXT = "Scene 's', decision 'd': hidden_modifiers";
const parse = (raw: unknown) => HiddenModifiers.parse(raw, CONTEXT);

const base = new Character({
  age: 30, workclass: 'Self-emp-not-inc', education: 'HS-grad', educationNum: 9, maritalStatus: 'Never-married',
  occupation: 'Craft-repair', relationship: 'Not-in-family', race: 'White', sex: 'Female', hoursPerWeek: 40,
  nativeCountry: 'Mexico', capitalGain: 1_000,
});

describe('HiddenModifiers.parse', () => {
  it.each([null, undefined, {}])('treats %s as no modifiers', (raw) => {
    expect(parse(raw)).toBe(HiddenModifiers.NONE);
    expect(parse(raw).isEmpty).toBe(true);
  });

  it('separates additive deltas from absolute replacements, accepting any key spelling', () => {
    const modifiers = parse({ 'capital-gain': 5000, age: 2, hours_per_week: -5, workclass: ' Private ', nativeCountry: 'Chile' });

    expect(modifiers.deltas).toEqual([['capitalGain', 5000], ['age', 2], ['hoursPerWeek', -5]]);
    expect(modifiers.replacements).toEqual([['workclass', 'Private'], ['nativeCountry', 'Chile']]);
  });

  it.each([
    ['an array', [1, 2], 'expected a JSON object'],
    ['a string', '{"age": 2}', 'expected a JSON object'],
    ['an unknown trait', { salary: 100 }, "'salary' is not a modifiable trait"],
    ['the census sampling weight', { fnlwgt: 100 }, "'fnlwgt' is not a modifiable trait"],
    ['a trait spelled twice', { 'capital-gain': 1, capitalGain: 2 }, "modifies 'capitalGain' a second time"],
    ['a fractional delta', { age: 1.5 }, 'must be an integer delta'],
    ['a string for a numeric trait', { age: '2' }, 'must be an integer delta'],
    ['a number for a categorical trait', { workclass: 3 }, 'must be a non-empty string'],
    ['an empty replacement', { workclass: '  ' }, 'must be a non-empty string'],
  ])('rejects %s', (_label, raw, message) => {
    expect(() => parse(raw)).toThrow(InvalidStoryContentError);
    expect(() => parse(raw)).toThrow(message);
  });

  it('names the scene and decision in the error', () => {
    expect(() => parse({ salary: 1 })).toThrow(`${CONTEXT}: 'salary'`);
  });
});

describe('injectContext', () => {
  it('returns the base character untouched when there are no modifiers', () => {
    const injection = injectContext(base, HiddenModifiers.NONE);

    expect(injection.character).toBe(base);
    expect(injection.adjustments).toEqual([]);
  });

  it('adds deltas and replaces categories on a copy, leaving the base profile intact', () => {
    const injection = injectContext(base, parse({ 'capital-gain': 5000, age: 2, workclass: 'Private' }));

    expect(injection.character).not.toBe(base);
    expect(injection.character).toMatchObject({ capitalGain: 6000, age: 32, workclass: 'Private', education: 'HS-grad' });
    expect(base).toMatchObject({ capitalGain: 1000, age: 30, workclass: 'Self-emp-not-inc' });
    expect(injection.adjustments).toEqual([
      { trait: 'capitalGain', operation: 'add', value: 5000, before: 1000, after: 6000, clamped: false },
      { trait: 'age', operation: 'add', value: 2, before: 30, after: 32, clamped: false },
      { trait: 'workclass', operation: 'set', value: 'Private', before: 'Self-emp-not-inc', after: 'Private', clamped: false },
    ]);
  });

  it.each([
    ['age above the training range', { age: 80 }, 'age', 90],
    ['hours below the minimum', { 'hours-per-week': -60 }, 'hoursPerWeek', 1],
    ['schooling above the maximum', { 'education-num': 20 }, 'educationNum', 16],
    ['a negative capital gain', { 'capital-gain': -5000 }, 'capitalGain', 0],
  ] as const)('projects %s onto the feasible range and records the clamp', (_label, raw, trait, expected) => {
    const injection = injectContext(base, parse(raw));

    expect(injection.character[trait]).toBe(expected);
    expect(injection.adjustments[0]).toMatchObject({ trait, after: expected, clamped: true });
  });

  it('produces a character that passes the entity invariants', () => {
    expect(injectContext(base, parse({ age: 5, sex: 'Male' })).character).toBeInstanceOf(Character);
  });
});
