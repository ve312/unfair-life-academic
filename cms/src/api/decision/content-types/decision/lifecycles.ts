/**
 * Decision lifecycle hooks: reject invalid hidden modifiers when an editor
 * saves, so that the error appears in the admin panel and the save is
 * aborted, instead of the scene failing to load in the BFF during play.
 */
import { errors } from '@strapi/utils';
import {
  findModifierIssues,
  HIDDEN_MODIFIERS_FIELD,
  MOBILITY_CHECK_FIELD,
} from './hidden-modifiers-validation';

const DECISION_UID = 'api::decision.decision';

type LifecycleEvent = {
  params: {
    data?: Record<string, unknown>;
    where?: Record<string, unknown>;
  };
};

function assertValidModifiers(values: Record<string, unknown>): void {
  const issues = findModifierIssues(values[HIDDEN_MODIFIERS_FIELD], values[MOBILITY_CHECK_FIELD] === true);
  if (issues.length > 0) {
    throw new errors.ValidationError(issues.map((issue) => issue.message).join(' '), {
      errors: issues.map((issue) => ({ path: [issue.path], message: issue.message, name: 'ValidationError' })),
    });
  }
}

/** True when the write touches the modifiers or the flag that enables them (other edits skip the check). */
function touchesModifiers(data: Record<string, unknown> | undefined): data is Record<string, unknown> {
  return data !== undefined && (HIDDEN_MODIFIERS_FIELD in data || MOBILITY_CHECK_FIELD in data);
}

export default {
  beforeCreate(event: LifecycleEvent) {
    const { data } = event.params;
    if (touchesModifiers(data)) {
      assertValidModifiers(data);
    }
  },

  async beforeUpdate(event: LifecycleEvent) {
    const { data, where } = event.params;
    if (!touchesModifiers(data)) {
      return;
    }
    // An update may change only one of the two fields: combine it with the stored other one,
    // so that unticking requires_mobility_check on a decision that keeps its modifiers is caught.
    const stored = where
      ? ((await strapi.db.query(DECISION_UID).findOne({
          where,
          select: [HIDDEN_MODIFIERS_FIELD, MOBILITY_CHECK_FIELD],
        })) as Record<string, unknown> | null)
      : null;
    assertValidModifiers({ ...(stored ?? {}), ...data });
  },
};
