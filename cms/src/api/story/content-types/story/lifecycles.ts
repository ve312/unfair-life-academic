/**
 * Story lifecycle hooks: reject an invalid base profile when an editor saves,
 * so that the error appears in the admin panel and the save is aborted,
 * instead of every mobility check of the story failing during play.
 */
import { errors } from '@strapi/utils';
import { BASE_PROFILE_FIELD, findProfileIssues } from './base-profile-validation';

type LifecycleEvent = {
  params: {
    data?: Record<string, unknown>;
  };
};

/** Validates the profile only when the write sets it (other edits, e.g. a new title, skip the check). */
function assertValidProfile(data: Record<string, unknown> | undefined): void {
  if (data === undefined || !(BASE_PROFILE_FIELD in data)) {
    return;
  }
  const issues = findProfileIssues(data[BASE_PROFILE_FIELD]);
  if (issues.length > 0) {
    throw new errors.ValidationError(issues.map((issue) => issue.message).join(' '), {
      errors: issues.map((issue) => ({ path: [issue.path], message: issue.message, name: 'ValidationError' })),
    });
  }
}

export default {
  beforeCreate(event: LifecycleEvent) {
    assertValidProfile(event.params.data);
  },

  beforeUpdate(event: LifecycleEvent) {
    assertValidProfile(event.params.data);
  },
};
