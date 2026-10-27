import { InvalidCharacterError, InvalidStoryContentError } from '../errors/domain.error.js';
import { AudioConfig } from './audio-config.entity.js';
import { Character, type CharacterProps } from './character.entity.js';
import { Decision } from './decision.entity.js';
import { HiddenModifiers } from './hidden-modifiers.entity.js';
import { StoryNode } from './story-node.entity.js';

const validCharacter: CharacterProps = {
  age: 34,
  workclass: ' Private ',
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

describe('Character', () => {
  it('accepts a valid profile, trims categories and applies defaults', () => {
    const character = new Character(validCharacter);
    expect(character.workclass).toBe('Private');
    expect(character.capitalGain).toBe(0);
    expect(character.capitalLoss).toBe(0);
    expect(character.fnlwgt).toBeNull();
    expect(Object.isFrozen(character)).toBe(true);
  });

  it.each([
    ['age below range', { age: 12 }],
    ['non-integer age', { age: 30.5 }],
    ['education level above range', { educationNum: 17 }],
    ['zero working hours', { hoursPerWeek: 0 }],
    ['blank category', { sex: '   ' }],
    ['negative capital gain', { capitalGain: -1 }],
    ['non-positive fnlwgt', { fnlwgt: 0 }],
  ])('rejects %s', (_label, override) => {
    expect(() => new Character({ ...validCharacter, ...override })).toThrow(InvalidCharacterError);
  });
});

describe('Decision', () => {
  it('always leads to the next scene when no mobility check is required', () => {
    const decision = new Decision({ id: 'd1', text: 'Start working', requiresMobilityCheck: false, nextSceneId: 's2' });
    expect(decision.resolveNextSceneId(false)).toBe('s2');
    expect(decision.resolveNextSceneId(true)).toBe('s2');
  });

  it('forks on a mobility check', () => {
    const decision = new Decision({
      id: 'd2',
      text: 'Apply to university',
      requiresMobilityCheck: true,
      nextSceneId: 'university',
      failureSceneId: 'factory',
    });
    expect(decision.resolveNextSceneId(true)).toBe('university');
    expect(decision.resolveNextSceneId(false)).toBe('factory');
  });

  it('rejects a mobility check without a failure scene', () => {
    expect(
      () => new Decision({ id: 'd3', text: 'Apply', requiresMobilityCheck: true, nextSceneId: 'university' }),
    ).toThrow(InvalidStoryContentError);
  });

  it('rejects a decision without a next scene', () => {
    expect(() => new Decision({ id: 'd4', text: 'Go', requiresMobilityCheck: false, nextSceneId: '' })).toThrow(
      InvalidStoryContentError,
    );
  });

  it('defaults to no hidden modifiers and keeps the ones of a mobility check', () => {
    const modifiers = HiddenModifiers.parse({ 'capital-gain': 5000 }, 'test');
    const plain = new Decision({ id: 'd5', text: 'Go', requiresMobilityCheck: false, nextSceneId: 'n', hiddenModifiers: null });
    const check = new Decision({
      id: 'd6', text: 'Borrow', requiresMobilityCheck: true, nextSceneId: 'n', failureSceneId: 'f', hiddenModifiers: modifiers,
    });

    expect(plain.hiddenModifiers).toBe(HiddenModifiers.NONE);
    expect(check.hiddenModifiers).toBe(modifiers);
  });

  it('rejects hidden modifiers on a decision without a mobility check, since they would never apply', () => {
    expect(
      () => new Decision({
        id: 'd7', text: 'Go', requiresMobilityCheck: false, nextSceneId: 'n',
        hiddenModifiers: HiddenModifiers.parse({ age: 2 }, 'test'),
      }),
    ).toThrow("Decision 'd7' declares hidden modifiers but no mobility check");
  });
});

describe('StoryNode', () => {
  const work = new Decision({ id: 'd1', text: 'Start working', requiresMobilityCheck: false, nextSceneId: 'factory' });

  it('builds a node and finds its decisions', () => {
    const node = new StoryNode({ id: 'intro', title: 'Intro', timerDuration: 10, isCinematic: false, decisions: [work] });
    expect(node.findDecision('d1')).toBe(work);
    expect(node.findDecision('missing')).toBeUndefined();
    expect(node.hasTimer).toBe(true);
    expect(node.isEnding).toBe(false);
    expect(node.backgroundMediaUrl).toBeNull();
    expect(node.audio).toBeNull();
    expect(node.isCinematic).toBe(false);
  });

  it('treats a node without decisions as an ending', () => {
    const node = new StoryNode({ id: 'end', title: 'The End', timerDuration: 0, isCinematic: true, decisions: [] });
    expect(node.isEnding).toBe(true);
    expect(node.hasTimer).toBe(false);
  });

  it.each([
    ['negative timer', { timerDuration: -1 }],
    ['timer above the CMS limit', { timerDuration: 601 }],
    ['blank title', { title: ' ' }],
    ['duplicate decisions', { decisions: [work, work] }],
    ['non-boolean isCinematic', { isCinematic: 'yes' as unknown as boolean }],
    ['audio that is not an AudioConfig', { audio: { url: '/rain.mp3', loop: true, defaultVolume: 1 } as AudioConfig }],
  ])('rejects %s', (_label, override) => {
    expect(
      () => new StoryNode({ id: 'intro', title: 'Intro', timerDuration: 10, isCinematic: false, decisions: [work], ...override }),
    ).toThrow(InvalidStoryContentError);
  });
});

describe('AudioConfig', () => {
  it('accepts a valid track and attaches it to a node', () => {
    const audio = new AudioConfig({ url: ' /uploads/rain.mp3 ', loop: true, defaultVolume: 0.6 });
    expect(audio.url).toBe('/uploads/rain.mp3');
    const node = new StoryNode({ id: 'intro', title: 'Intro', timerDuration: 0, isCinematic: true, decisions: [], audio });
    expect(node.audio).toBe(audio);
  });

  it.each([
    ['blank url', { url: '' }],
    ['non-boolean loop', { loop: 'yes' as unknown as boolean }],
    ['volume above 1', { defaultVolume: 1.5 }],
    ['negative volume', { defaultVolume: -0.1 }],
    ['NaN volume', { defaultVolume: Number.NaN }],
  ])('rejects %s', (_label, override) => {
    expect(() => new AudioConfig({ url: '/uploads/rain.mp3', loop: true, defaultVolume: 0.6, ...override })).toThrow(
      InvalidStoryContentError,
    );
  });
});
