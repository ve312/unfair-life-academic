import { InvalidStoryContentError } from '../errors/domain.error.js';
import { isSupportedStoryYear, STORY_YEAR_RANGE } from '../services/consumer-price-index.js';

/** Matches the CMS schema limit for Story.short_description. */
export const MAX_SHORT_DESCRIPTION_LENGTH = 280;

export interface StorySummaryProps {
  id: string;
  title: string;
  shortDescription?: string | null;
  coverImageUrl?: string | null;
  year?: number | null;
}

/**
 * A playable story as shown in the story-selection menu: enough to choose it,
 * nothing about its scenes or decisions.
 */
export class StorySummary {
  readonly id: string;
  readonly title: string;
  readonly shortDescription: string | null;
  readonly coverImageUrl: string | null;
  readonly year: number | null;

  constructor(props: StorySummaryProps) {
    this.id = StorySummary.requiredText('id', props.id);
    this.title = StorySummary.requiredText('title', props.title, this.id);
    this.shortDescription = StorySummary.optionalText('shortDescription', props.shortDescription, this.id);
    if (this.shortDescription !== null && this.shortDescription.length > MAX_SHORT_DESCRIPTION_LENGTH) {
      throw new InvalidStoryContentError(
        `Story '${this.id}': 'shortDescription' must be at most ${MAX_SHORT_DESCRIPTION_LENGTH} characters`,
      );
    }
    this.coverImageUrl = StorySummary.optionalText('coverImageUrl', props.coverImageUrl, this.id);
    if (props.year != null && !isSupportedStoryYear(props.year)) {
      throw new InvalidStoryContentError(
        `Story '${this.id}': 'year' must be an integer between ${STORY_YEAR_RANGE.min} and ${STORY_YEAR_RANGE.max}`,
      );
    }
    this.year = props.year ?? null;
    Object.freeze(this);
  }

  private static requiredText(name: string, value: unknown, storyId?: string): string {
    if (typeof value !== 'string' || value.trim().length === 0) {
      const owner = storyId ? `Story '${storyId}': ` : 'Story: ';
      throw new InvalidStoryContentError(`${owner}'${name}' must be a non-empty string`);
    }
    return value.trim();
  }

  /** Empty or whitespace-only text is treated as absent. */
  private static optionalText(name: string, value: unknown, storyId: string): string | null {
    if (value == null) {
      return null;
    }
    if (typeof value !== 'string') {
      throw new InvalidStoryContentError(`Story '${storyId}': '${name}' must be a string`);
    }
    const trimmed = value.trim();
    return trimmed.length === 0 ? null : trimmed;
  }
}
