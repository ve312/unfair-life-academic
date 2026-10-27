import { Inject, Injectable, Logger } from '@nestjs/common';
import type { StoryNode } from '../../../core/domain/entities/story-node.entity.js';
import type { StorySummary } from '../../../core/domain/entities/story-summary.entity.js';
import {
  InvalidStoryContentError,
  StoryNodeNotFoundError,
  StoryNotFoundError,
} from '../../../core/domain/errors/domain.error.js';
import type { StoryContentPort } from '../../../core/ports/out/story-content.port.js';
import { STRAPI_CONFIG, type StrapiConfig } from '../../config/external-services.config.js';
import { ExternalServiceError } from '../external-service.error.js';
import { describeErrorBody, requestJson } from '../http/request-json.js';
import { mapStartingNode, mapStoryNode, mapStorySummary, unwrapStrapiData, unwrapStrapiList } from './strapi-mappers.js';

const SERVICE = 'cms';

type QueryValue = string | number | boolean | QueryValue[] | { [key: string]: QueryValue };

/**
 * Explicit populate for one scene: only the fields the game needs, two levels
 * deep (scene -> decisions -> target scene ids, scene -> audio track -> file).
 * Decisions are sorted by creation order so choices appear in authoring order.
 */
const SCENE_QUERY = {
  fields: [
    'title', 'context', 'timer_duration', 'is_cinematic', 'year',
    'media_start_time', 'media_end_time', 'audio_start_time', 'audio_end_time',
  ],
  populate: {
    // The scene's story year is the fallback when the scene has no year of its own.
    // The story supplies the fallback year and the base profile of the character (server-side only).
    story: { fields: ['year', 'base_profile'] },
    background_media: { fields: ['url'] },
    audio_track: {
      fields: ['loop', 'default_volume'],
      populate: { audio_file: { fields: ['url'] } },
    },
    available_decisions: {
      // hidden_modifiers stays inside the BFF: the presenter never sends it to the game client.
      fields: ['text', 'requires_mobility_check', 'hidden_modifiers'],
      sort: ['id:asc'],
      populate: {
        next_scene: { fields: ['title'] },
        failure_scene: { fields: ['title'] },
      },
    },
  },
} satisfies QueryValue;

const STORY_QUERY = {
  fields: ['title', 'year', 'base_profile'],
  populate: { starting_scene: SCENE_QUERY },
} satisfies QueryValue;

/**
 * Menu listing: only stories that can actually be started (they have a
 * starting scene), alphabetically, with just the fields a menu card needs.
 */
const STORY_LIST_QUERY = {
  fields: ['title', 'short_description', 'year'],
  populate: { cover_image: { fields: ['url'] } },
  filters: { starting_scene: { id: { $notNull: true } } },
  sort: ['title:asc'],
} satisfies QueryValue;

/** Strapi's maximum page size; larger catalogs are fetched page by page. */
const STORY_LIST_PAGE_SIZE = 100;
/** Hard stop so a misbehaving CMS cannot make the BFF page forever. */
const MAX_STORY_LIST_PAGES = 20;

/**
 * Serialises a nested query object into Strapi's bracket notation, e.g.
 * { populate: { audio_track: { fields: ['loop'] } } } ->
 * populate[audio_track][fields][0]=loop
 */
export function toStrapiQueryString(query: QueryValue): string {
  const params = new URLSearchParams();
  const visit = (value: QueryValue, key: string): void => {
    if (Array.isArray(value)) {
      value.forEach((item, index) => visit(item, `${key}[${index}]`));
    } else if (typeof value === 'object') {
      Object.entries(value).forEach(([childKey, child]) => visit(child, key ? `${key}[${childKey}]` : childKey));
    } else {
      params.append(key, String(value));
    }
  };
  visit(query, '');
  return params.toString();
}

/** StoryContentPort backed by the Strapi CMS REST API, guarded by the is-bff-server policy. */
@Injectable()
export class StrapiContentAdapter implements StoryContentPort {
  private readonly logger = new Logger(StrapiContentAdapter.name);

  constructor(@Inject(STRAPI_CONFIG) private readonly config: StrapiConfig) {}

  async getStoryNode(nodeId: string): Promise<StoryNode> {
    if (nodeId.trim().length === 0) {
      throw new StoryNodeNotFoundError(nodeId);
    }
    const body = await this.getDocument('scenes', nodeId, SCENE_QUERY, () => new StoryNodeNotFoundError(nodeId));
    return mapStoryNode(unwrapStrapiData(body, `Scene '${nodeId}' response`), this.config.mediaBaseUrl);
  }

  async getStartingNode(storyId: string): Promise<StoryNode> {
    if (storyId.trim().length === 0) {
      throw new StoryNotFoundError(storyId);
    }
    const body = await this.getDocument('stories', storyId, STORY_QUERY, () => new StoryNotFoundError(storyId));
    return mapStartingNode(unwrapStrapiData(body, `Story '${storyId}' response`), this.config.mediaBaseUrl);
  }

  async listStories(): Promise<StorySummary[]> {
    const stories: StorySummary[] = [];
    for (let page = 1; page <= MAX_STORY_LIST_PAGES; page++) {
      const body = await this.get('/api/stories', { ...STORY_LIST_QUERY, pagination: { page, pageSize: STORY_LIST_PAGE_SIZE } });
      const { items, pageCount } = unwrapStrapiList(body, 'Story list response');
      for (const raw of items) {
        try {
          stories.push(mapStorySummary(raw, this.config.mediaBaseUrl));
        } catch (error) {
          if (!(error instanceof InvalidStoryContentError)) {
            throw error;
          }
          // One malformed story must not hide every other story from the menu.
          this.logger.warn(`Skipping story in the menu: ${error.message}`);
        }
      }
      if (page >= pageCount) {
        return stories;
      }
    }
    this.logger.warn(`Story list truncated at ${MAX_STORY_LIST_PAGES * STORY_LIST_PAGE_SIZE} entries`);
    return stories;
  }

  private async getDocument(
    collection: 'scenes' | 'stories',
    documentId: string,
    query: QueryValue,
    notFound: () => Error,
  ): Promise<unknown> {
    return this.get(`/api/${collection}/${encodeURIComponent(documentId)}`, query, notFound);
  }

  /** GET with the BFF token; maps 404 to `notFound` and auth / server failures to ExternalServiceError. */
  private async get(path: string, query: QueryValue, notFound?: () => Error): Promise<unknown> {
    const { status, body } = await requestJson(SERVICE, `${this.config.baseUrl}${path}?${toStrapiQueryString(query)}`, {
      method: 'GET',
      // Exact header format required by the CMS policy: the bare token, no "Bearer" prefix.
      headers: { Authorization: this.config.token },
      timeoutMs: this.config.timeoutMs,
    });

    if (status === 200) {
      return body;
    }
    if (status === 404 && notFound) {
      throw notFound();
    }
    if (status === 401 || status === 403) {
      this.logger.error(`CMS rejected GET ${path} with ${status}; check that BFF_SECRET_TOKEN matches the CMS .env`);
      throw new ExternalServiceError(SERVICE, `GET ${path} was rejected (${status}): the BFF token is not accepted`, {
        status,
      });
    }
    throw new ExternalServiceError(SERVICE, `GET ${path} responded ${status}: ${describeErrorBody(body)}`, { status });
  }
}
