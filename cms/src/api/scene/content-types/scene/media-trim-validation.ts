/**
 * Validation of a scene's media fragments (start / end times in seconds).
 *
 * Mirrors the BFF rule (bff-api/src/core/domain/entities/media-trim.entity.ts)
 * so anything the admin panel accepts can always be played: each bound is
 * empty or a number between 0 and 24 hours, and an end time must be strictly
 * after its start (an empty start counts as 0).
 */

/** Same upper bound as the BFF: 24 hours, which catches milliseconds typed as seconds. */
export const MAX_MEDIA_TIME_SECONDS = 86_400;

export const TRIM_PAIRS = [
  { label: 'Background media', start: 'media_start_time', end: 'media_end_time' },
  { label: 'Audio', start: 'audio_start_time', end: 'audio_end_time' },
] as const;

export const TRIM_FIELDS: readonly string[] = TRIM_PAIRS.flatMap((pair) => [pair.start, pair.end]);

export interface TrimIssue {
  path: string;
  message: string;
}

type Values = Record<string, unknown>;

/** A bound as a number, null when empty, or undefined when it is not a valid number. */
function toSeconds(value: unknown): number | null | undefined {
  if (value === null || value === undefined || value === '') {
    return null;
  }
  const parsed = typeof value === 'string' ? Number(value) : value;
  return typeof parsed === 'number' && Number.isFinite(parsed) ? parsed : undefined;
}

/** Returns every problem with the trims in `values` (empty when they are valid). */
export function findTrimIssues(values: Values): TrimIssue[] {
  const issues: TrimIssue[] = [];
  for (const pair of TRIM_PAIRS) {
    const start = toSeconds(values[pair.start]);
    const end = toSeconds(values[pair.end]);

    for (const [field, seconds] of [[pair.start, start], [pair.end, end]] as const) {
      if (seconds === undefined) {
        issues.push({ path: field, message: `${pair.label}: ${field} must be a number of seconds` });
      } else if (seconds !== null && (seconds < 0 || seconds > MAX_MEDIA_TIME_SECONDS)) {
        issues.push({ path: field, message: `${pair.label}: ${field} must be between 0 and ${MAX_MEDIA_TIME_SECONDS} seconds` });
      }
    }

    if (typeof end === 'number' && start !== undefined && end <= (start ?? 0)) {
      issues.push({
        path: pair.end,
        message: `${pair.label}: the end time (${end}s) must be after the start time (${start ?? 0}s)`,
      });
    }
  }
  return issues;
}
