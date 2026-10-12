import type { StorySummary } from '../../../core/domain/entities/story-summary.entity.js';

/** One entry of GET /game/stories, as consumed by the story-selection menu. */
export interface StorySummaryDto {
  /** Strapi documentId; pass it to POST /game/start/:storyId. */
  id: string;
  title: string;
  shortDescription: string | null;
  /** Absolute URL the client can load, or null when the story has no cover. */
  coverImageUrl: string | null;
  /** Year the story is set in, or null when the content has none. */
  year: number | null;
}

export function toStorySummaryDto(summary: StorySummary): StorySummaryDto {
  return {
    id: summary.id,
    title: summary.title,
    shortDescription: summary.shortDescription,
    coverImageUrl: summary.coverImageUrl,
    year: summary.year,
  };
}
