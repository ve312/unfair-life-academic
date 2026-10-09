import { AudioConfig } from '../../../core/domain/entities/audio-config.entity.js';
import { Character } from '../../../core/domain/entities/character.entity.js';
import { Decision } from '../../../core/domain/entities/decision.entity.js';
import { HiddenModifiers } from '../../../core/domain/entities/hidden-modifiers.entity.js';
import { MediaTrim } from '../../../core/domain/entities/media-trim.entity.js';
import { StoryNode } from '../../../core/domain/entities/story-node.entity.js';
import { DEFAULT_STORY_YEAR } from '../../../core/use-cases/resolve-decision.service.js';
import { presentStoryNode } from './story-node.presenter.js';

const node = (year: number | null) =>
  new StoryNode({
    id: 'intro', title: 'Intro', timerDuration: 10, isCinematic: false, year,
    decisions: [new Decision({ id: 'apply', text: 'Apply', requiresMobilityCheck: true, nextSceneId: 'uni', failureSceneId: 'farm' })],
  });

describe('presentStoryNode', () => {
  it("reports what mobility means in the scene's year", () => {
    expect(presentStoryNode(node(2024)).economy).toEqual({ year: 2024, mobilityIncomeThreshold: 105_833, cpiPublished: true });
    expect(presentStoryNode(node(2026)).economy).toEqual({ year: 2026, mobilityIncomeThreshold: 111_191, cpiPublished: false });
  });

  it('uses the default story year when the content has none', () => {
    expect(presentStoryNode(node(null)).economy.year).toBe(DEFAULT_STORY_YEAR);
  });

  it('never reveals where a decision leads', () => {
    const view = presentStoryNode(node(2026));
    expect(view.decisions).toEqual([{ id: 'apply', text: 'Apply', requiresMobilityCheck: true }]);
    expect(JSON.stringify(view)).not.toMatch(/uni|farm/);
  });

  it('sends the narrative context, or null when the scene has none', () => {
    const withContext = new StoryNode({
      id: 'intro', title: 'Intro', context: 'Ana is 19 and needs a job.', timerDuration: 0, isCinematic: false, decisions: [],
    });

    expect(presentStoryNode(withContext).context).toBe('Ana is 19 and needs a job.');
    expect(presentStoryNode(node(2026)).context).toBeNull();
  });

  it("never sends the story's base profile to the client", () => {
    const withProfile = new StoryNode({
      id: 'intro', title: 'Intro', timerDuration: 0, isCinematic: false, decisions: [],
      baseProfile: new Character({
        age: 19, workclass: 'Private', education: 'HS-grad', educationNum: 9, maritalStatus: 'Never-married',
        occupation: 'Farming-fishing', relationship: 'Own-child', race: 'Amer-Indian-Eskimo', sex: 'Male', hoursPerWeek: 50,
        nativeCountry: 'Guatemala',
      }),
    });

    expect(JSON.stringify(presentStoryNode(withProfile))).not.toMatch(/baseProfile|Guatemala|Farming/);
  });

  it('never reveals the hidden modifiers of a decision', () => {
    const withModifiers = new StoryNode({
      id: 'intro', title: 'Intro', timerDuration: 10, isCinematic: false,
      decisions: [new Decision({
        id: 'loan', text: 'Take the loan', requiresMobilityCheck: true, nextSceneId: 'uni', failureSceneId: 'farm',
        hiddenModifiers: HiddenModifiers.parse({ 'capital-gain': 5000, workclass: 'Private' }, 'test'),
      })],
    });

    const view = presentStoryNode(withModifiers);

    expect(view.decisions).toEqual([{ id: 'loan', text: 'Take the loan', requiresMobilityCheck: true }]);
    expect(JSON.stringify(view)).not.toMatch(/5000|Private|modifier|capital/i);
  });
});

describe('presentStoryNode media trims', () => {
  it('exposes the background fragment and the audio fragment', () => {
    const node = new StoryNode({
      id: 'clip', title: 'Clip', timerDuration: 0, isCinematic: true, decisions: [],
      backgroundMediaUrl: 'http://cms/uploads/long.mp4',
      backgroundTrim: new MediaTrim({ startTime: 62, endTime: 75.5 }),
      audio: new AudioConfig({ url: 'http://cms/uploads/song.mp3', loop: true, defaultVolume: 0.8 }),
      audioTrim: new MediaTrim({ startTime: 30 }),
    });

    const view = presentStoryNode(node);

    expect(view).toMatchObject({ mediaStartTime: 62, mediaEndTime: 75.5 });
    expect(view.audio).toEqual({ url: 'http://cms/uploads/song.mp3', loop: true, defaultVolume: 0.8, startTime: 30, endTime: null });
  });

  it('sends nulls when the scene plays whole files', () => {
    const view = presentStoryNode(node(2026));

    expect(view).toMatchObject({ mediaStartTime: null, mediaEndTime: null });
  });
});
