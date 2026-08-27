import { InvalidStoryContentError } from '../errors/domain.error.js';
import { parseBaseProfile } from './base-profile.js';
import { Character } from './character.entity.js';
import { HiddenModifiers } from './hidden-modifiers.entity.js';

const CONTEXT = "Scene 's': story base_profile";

/** The default profile of a new story in the CMS (kebab-case keys). */
const cmsDefault = () => ({
  age: 25, workclass: 'Private', education: 'Bachelors', 'education-num': 13, 'marital-status': 'Never-married',
  occupation: 'Prof-specialty', relationship: 'Not-in-family', race: 'White', sex: 'Male', 'hours-per-week': 40,
  'native-country': 'United-States', 'capital-gain': 0, 'capital-loss': 0,
});

describe('parseBaseProfile', () => {
  it('builds a Character from the CMS default profile, whatever the key spelling', () => {
    const profile = parseBaseProfile(cmsDefault(), CONTEXT);

    expect(profile).toBeInstanceOf(Character);
    expect(profile).toMatchObject({ age: 25, educationNum: 13, maritalStatus: 'Never-married', nativeCountry: 'United-States' });
    const { 'capital-gain': _kebab, ...rest } = cmsDefault();
    expect(parseBaseProfile({ ...rest, capital_gain: 7000 }, CONTEXT)?.capitalGain).toBe(7000);
  });

  it.each([null, undefined, {}])('returns null for a story without a profile (%j)', (raw) => {
    expect(parseBaseProfile(raw, CONTEXT)).toBeNull();
  });

  it.each([
    ['an array', [1], 'expected a JSON object'],
    ['a string', '{"age": 25}', 'expected a JSON object'],
    ['an unknown trait', { ...cmsDefault(), salary: 1 }, "'salary' is not a trait of the model"],
    ['the census sampling weight', { ...cmsDefault(), fnlwgt: 100 }, "'fnlwgt' is not a trait of the model"],
    ['a trait spelled twice', { ...cmsDefault(), capitalGain: 5 }, "'capitalGain' defines 'capitalGain' a second time"],
    ['a missing trait', { age: 30, sex: 'Female' }, 'missing traits educationNum'],
    ['an out-of-range age', { ...cmsDefault(), age: 12 }, "'age' must be between 17 and 90"],
  ])('rejects %s, naming the story', (_label, raw, message) => {
    expect(() => parseBaseProfile(raw, CONTEXT)).toThrow(InvalidStoryContentError);
    expect(() => parseBaseProfile(raw, CONTEXT)).toThrow(`${CONTEXT}: ${message}`);
  });
});

describe('HiddenModifiers zero deltas', () => {
  it('treats the CMS zero template as no modifiers (identity of addition)', () => {
    expect(HiddenModifiers.parse({ 'capital-gain': 0, 'capital-loss': 0, age: 0 }, 'test')).toBe(HiddenModifiers.NONE);
  });

  it('keeps the non-zero deltas and drops the zero ones', () => {
    expect(HiddenModifiers.parse({ 'capital-gain': 5000, 'capital-loss': 0, age: 0 }, 'test').deltas).toEqual([['capitalGain', 5000]]);
  });
});
