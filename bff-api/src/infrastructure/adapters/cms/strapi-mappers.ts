import { AudioConfig } from '../../../core/domain/entities/audio-config.entity.js';
import { parseBaseProfile } from '../../../core/domain/entities/base-profile.js';
import type { Character } from '../../../core/domain/entities/character.entity.js';
import { Decision } from '../../../core/domain/entities/decision.entity.js';
import { HiddenModifiers } from '../../../core/domain/entities/hidden-modifiers.entity.js';
import { MediaTrim } from '../../../core/domain/entities/media-trim.entity.js';
import { StoryNode } from '../../../core/domain/entities/story-node.entity.js';
import { StorySummary } from '../../../core/domain/entities/story-summary.entity.js';
import { InvalidStoryContentError } from '../../../core/domain/errors/domain.error.js';

/**
 * Turns Strapi v5 REST payloads into domain entities.
 *
 * Strapi v5 returns flat entries ({ documentId, title, ... }) with populated
 * relations nested as objects or arrays, and media as { url, ... }. Payloads
 * are treated as untrusted: every field is checked, and any mismatch raises
 * InvalidStoryContentError naming the offending scene or decision, so broken
 * content surfaces at load time instead of mid-game.
 */

type JsonRecord = Record<string, unknown>;

/** Unwraps the { data, meta } envelope of a Strapi findOne response. */
export function unwrapStrapiData(body: unknown, context: string): JsonRecord {
  const envelope = asRecord(body, context);
  return asRecord(envelope.data, `${context}: 'data'`);
}

/**
 * Unwraps a Strapi find (list) response: { data: [...], meta: { pagination } }.
 * A missing pagination block is treated as a single page.
 */
export function unwrapStrapiList(body: unknown, context: string): { items: unknown[]; pageCount: number } {
  const envelope = asRecord(body, context);
  if (!Array.isArray(envelope.data)) {
    throw new InvalidStoryContentError(`${context}: 'data' must be an array`);
  }
  const meta = envelope.meta as JsonRecord | undefined;
  const pagination = meta?.pagination as JsonRecord | undefined;
  const pageCount = pagination?.pageCount;
  return { items: envelope.data, pageCount: typeof pageCount === 'number' && Number.isInteger(pageCount) ? pageCount : 1 };
}

/** Maps a Strapi story list entry to the summary shown in the story menu. */
export function mapStorySummary(rawStory: unknown, mediaBaseUrl: string): StorySummary {
  const story = asRecord(rawStory, 'Story list entry');
  const id = requireString(story.documentId, 'Story list entry: documentId');
  const context = `Story '${id}'`;
  return new StorySummary({
    id,
    title: requireString(story.title, `${context}: title`),
    shortDescription: story.short_description as string | null | undefined,
    coverImageUrl: mapMediaUrl(story.cover_image, mediaBaseUrl, `${context}: cover_image`),
    year: optionalYear(story.year, `${context}: year`),
  });
}

/**
 * Maps a populated Strapi scene to a StoryNode. The year is the scene's own,
 * else its story's, else `fallbackYear` (the story being started), else null.
 * The base profile is the scene's story's, else `fallbackProfile` (the raw
 * base_profile of the story being started), else null.
 */
export function mapStoryNode(
  rawScene: unknown,
  mediaBaseUrl: string,
  fallbackYear: number | null = null,
  fallbackProfile: unknown = null,
): StoryNode {
  const scene = asRecord(rawScene, 'Scene');
  const id = requireString(scene.documentId, 'Scene: documentId');
  const context = `Scene '${id}'`;

  const rawDecisions = scene.available_decisions;
  if (!Array.isArray(rawDecisions)) {
    throw new InvalidStoryContentError(`${context}: 'available_decisions' was not populated`);
  }

  return new StoryNode({
    id,
    title: requireString(scene.title, `${context}: title`),
    context: scene.context as string | null | undefined,
    backgroundMediaUrl: mapMediaUrl(scene.background_media, mediaBaseUrl, `${context}: background_media`),
    audio: mapAudioConfig(scene.audio_track, mediaBaseUrl, `${context}: audio_track`),
    backgroundTrim: mapMediaTrim(scene.media_start_time, scene.media_end_time, `${context}: media trim`),
    audioTrim: mapMediaTrim(scene.audio_start_time, scene.audio_end_time, `${context}: audio trim`),
    timerDuration: optionalInteger(scene.timer_duration, 0, `${context}: timer_duration`),
    isCinematic: optionalBoolean(scene.is_cinematic, false, `${context}: is_cinematic`),
    year: optionalYear(scene.year, `${context}: year`) ?? storyYear(scene.story, context) ?? fallbackYear,
    baseProfile: storyProfile(scene.story, context) ?? parseBaseProfile(fallbackProfile, `${context}: story base_profile`),
    decisions: rawDecisions.map((rawDecision) => mapDecision(rawDecision, id)),
  });
}

/** Maps a populated Strapi decision to a Decision. */
export function mapDecision(rawDecision: unknown, sceneId: string): Decision {
  const decision = asRecord(rawDecision, `Scene '${sceneId}': decision`);
  const id = requireString(decision.documentId, `Scene '${sceneId}': decision documentId`);
  const context = `Scene '${sceneId}', decision '${id}'`;

  const nextScene = optionalRelationId(decision.next_scene, `${context}: next_scene`);
  if (nextScene === null) {
    throw new InvalidStoryContentError(`${context}: 'next_scene' is not set`);
  }

  return new Decision({
    id,
    text: requireString(decision.text, `${context}: text`),
    requiresMobilityCheck: optionalBoolean(decision.requires_mobility_check, false, `${context}: requires_mobility_check`),
    nextSceneId: nextScene,
    failureSceneId: optionalRelationId(decision.failure_scene, `${context}: failure_scene`),
    // Strapi JSON field; null when the editor left it empty, which means "no modifiers".
    hiddenModifiers: HiddenModifiers.parse(decision.hidden_modifiers, `${context}: hidden_modifiers`),
  });
}

/** Maps a Strapi story with a populated starting_scene to its first StoryNode. */
export function mapStartingNode(rawStory: unknown, mediaBaseUrl: string): StoryNode {
  const story = asRecord(rawStory, 'Story');
  const id = requireString(story.documentId, 'Story: documentId');
  if (story.starting_scene === null || story.starting_scene === undefined) {
    throw new InvalidStoryContentError(`Story '${id}' has no starting scene`);
  }
  return mapStoryNode(story.starting_scene, mediaBaseUrl, optionalYear(story.year, `Story '${id}': year`), story.base_profile);
}

/** Maps a populated audio track to an AudioConfig; null when the scene has no track. */
export function mapAudioConfig(rawTrack: unknown, mediaBaseUrl: string, context: string): AudioConfig | null {
  if (rawTrack === null || rawTrack === undefined) {
    return null;
  }
  const track = asRecord(rawTrack, context);
  const url = mapMediaUrl(track.audio_file, mediaBaseUrl, `${context}: audio_file`);
  if (url === null) {
    throw new InvalidStoryContentError(`${context}: the audio track has no audio file`);
  }
  return new AudioConfig({
    url,
    loop: optionalBoolean(track.loop, false, `${context}: loop`),
    defaultVolume: parseDecimal(track.default_volume, 1, `${context}: default_volume`),
  });
}

/**
 * Returns an absolute URL for a Strapi media object, or null when unset.
 * The local upload provider stores paths like '/uploads/rain.mp3'; cloud
 * providers store absolute URLs, which are kept as they are.
 */
export function mapMediaUrl(rawMedia: unknown, mediaBaseUrl: string, context: string): string | null {
  if (rawMedia === null || rawMedia === undefined) {
    return null;
  }
  const media = asRecord(rawMedia, context);
  const url = requireString(media.url, `${context}: url`);
  try {
    return new URL(url, `${mediaBaseUrl}/`).toString();
  } catch {
    throw new InvalidStoryContentError(`${context}: '${url}' is not a valid media URL`);
  }
}

/** A year field: null when unset, an integer otherwise (range is checked by StoryNode). */
function optionalYear(value: unknown, context: string): number | null {
  if (value === null || value === undefined) {
    return null;
  }
  if (typeof value !== 'number' || !Number.isInteger(value)) {
    throw new InvalidStoryContentError(`${context}: expected an integer year`);
  }
  return value;
}

function storyYear(rawStory: unknown, sceneContext: string): number | null {
  if (rawStory === null || rawStory === undefined) {
    return null;
  }
  return optionalYear(asRecord(rawStory, `${sceneContext}: story`).year, `${sceneContext}: story year`);
}

/** The base profile of the scene's story; null when the scene has no story or the story declares none. */
function storyProfile(rawStory: unknown, sceneContext: string): Character | null {
  if (rawStory === null || rawStory === undefined) {
    return null;
  }
  return parseBaseProfile(asRecord(rawStory, `${sceneContext}: story`).base_profile, `${sceneContext}: story base_profile`);
}

function optionalRelationId(rawRelation: unknown, context: string): string | null {
  if (rawRelation === null || rawRelation === undefined) {
    return null;
  }
  return requireString(asRecord(rawRelation, context).documentId, `${context}: documentId`);
}

function asRecord(value: unknown, context: string): JsonRecord {
  if (value === null || typeof value !== 'object' || Array.isArray(value)) {
    throw new InvalidStoryContentError(`${context}: expected an object`);
  }
  return value as JsonRecord;
}

function requireString(value: unknown, context: string): string {
  if (typeof value !== 'string' || value.trim().length === 0) {
    throw new InvalidStoryContentError(`${context}: expected a non-empty string`);
  }
  return value;
}

function optionalBoolean(value: unknown, fallback: boolean, context: string): boolean {
  if (value === null || value === undefined) {
    return fallback;
  }
  if (typeof value !== 'boolean') {
    throw new InvalidStoryContentError(`${context}: expected a boolean`);
  }
  return value;
}

function optionalInteger(value: unknown, fallback: number, context: string): number {
  if (value === null || value === undefined) {
    return fallback;
  }
  if (typeof value !== 'number' || !Number.isInteger(value)) {
    throw new InvalidStoryContentError(`${context}: expected an integer`);
  }
  return value;
}

/** Builds a trim from two optional second values (floats may arrive as strings on PostgreSQL). */
export function mapMediaTrim(rawStart: unknown, rawEnd: unknown, context: string): MediaTrim {
  return new MediaTrim(
    {
      startTime: optionalSeconds(rawStart, `${context}: start time`),
      endTime: optionalSeconds(rawEnd, `${context}: end time`),
    },
    context,
  );
}

function optionalSeconds(value: unknown, context: string): number | null {
  if (value === null || value === undefined) {
    return null;
  }
  const parsed = typeof value === 'string' && value.trim() !== '' ? Number(value) : value;
  if (typeof parsed !== 'number' || !Number.isFinite(parsed)) {
    throw new InvalidStoryContentError(`${context}: expected a number of seconds`);
  }
  return parsed;
}

/** Decimals arrive as numbers on SQLite but as strings on PostgreSQL. */
function parseDecimal(value: unknown, fallback: number, context: string): number {
  if (value === null || value === undefined) {
    return fallback;
  }
  const parsed = typeof value === 'string' && value.trim() !== '' ? Number(value) : value;
  if (typeof parsed !== 'number' || !Number.isFinite(parsed)) {
    throw new InvalidStoryContentError(`${context}: expected a number`);
  }
  return parsed;
}
