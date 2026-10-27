import { DomainError } from '../errors/domain.error.js';
import type { MobilityAssessment } from '../entities/mobility-assessment.entity.js';
import type { RandomSource } from '../../ports/out/random-source.port.js';

/** The random source produced a value outside [0, 1). */
export class InvalidRandomValueError extends DomainError {}

export interface MobilityOutcome {
  hasMobility: boolean;
  /** 'model' when the AI was confident, 'luck' when the weighted coin flip decided. */
  decidedBy: 'model' | 'luck';
  /** The uniform draw in [0, 1) used inside the band; null when the model decided. */
  roll: number | null;
  assessment: MobilityAssessment;
}

/**
 * Systemic luck: the extremes of society are deterministic, the middle is left to chance.
 *
 *   zone 'mobility'    -> mobility                       (no roll)
 *   zone 'no_mobility' -> no mobility                    (no roll)
 *   zone 'uncertain'   -> mobility iff roll < probability, roll ~ U[0, 1)
 *
 * Inside the band the chance of mobility therefore equals the model's
 * (calibrated) probability: a 60% character succeeds 60% of the time.
 */
export function resolveMobility(assessment: MobilityAssessment, random: RandomSource): MobilityOutcome {
  if (assessment.zone === 'mobility' || assessment.zone === 'no_mobility') {
    return { hasMobility: assessment.zone === 'mobility', decidedBy: 'model', roll: null, assessment };
  }
  const roll = random.next();
  if (typeof roll !== 'number' || !Number.isFinite(roll) || roll < 0 || roll >= 1) {
    throw new InvalidRandomValueError(`Random source must return a number in [0, 1), got ${String(roll)}`);
  }
  return { hasMobility: roll < assessment.probability, decidedBy: 'luck', roll, assessment };
}
