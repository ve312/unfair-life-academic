/**
 * Scene lifecycle hooks: reject invalid media fragments when an editor saves,
 * so the error appears in the admin panel instead of as a 502 during play.
 */
import { errors } from '@strapi/utils';
import { findTrimIssues, TRIM_FIELDS } from './media-trim-validation';

const SCENE_UID = 'api::scene.scene';

type LifecycleEvent = {
  params: {
    data?: Record<string, unknown>;
    where?: Record<string, unknown>;
  };
};

function assertValidTrims(values: Record<string, unknown>): void {
  const issues = findTrimIssues(values);
  if (issues.length > 0) {
    throw new errors.ValidationError(issues.map((issue) => issue.message).join('; '), {
      errors: issues.map((issue) => ({ path: [issue.path], message: issue.message, name: 'ValidationError' })),
    });
  }
}

/** True when the write touches at least one trim field (other edits skip the check). */
function touchesTrims(data: Record<string, unknown> | undefined): data is Record<string, unknown> {
  return data !== undefined && TRIM_FIELDS.some((field) => field in data);
}

export default {
  beforeCreate(event: LifecycleEvent) {
    const { data } = event.params;
    if (touchesTrims(data)) {
      assertValidTrims(data);
    }
  },

  async beforeUpdate(event: LifecycleEvent) {
    const { data, where } = event.params;
    if (!touchesTrims(data)) {
      return;
    }
    // An update may change only one bound: compare it with the stored other bound.
    const stored = where
      ? ((await strapi.db.query(SCENE_UID).findOne({ where, select: [...TRIM_FIELDS] })) as Record<string, unknown> | null)
      : null;
    assertValidTrims({ ...(stored ?? {}), ...data });
  },
};
