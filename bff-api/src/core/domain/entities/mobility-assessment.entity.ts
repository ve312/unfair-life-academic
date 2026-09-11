import { DomainError } from '../errors/domain.error.js';

/** Where the model's probability falls relative to its uncertainty band. */
export type MobilityZone = 'mobility' | 'no_mobility' | 'uncertain';

export const MOBILITY_ZONES: readonly MobilityZone[] = Object.freeze(['mobility', 'no_mobility', 'uncertain']);

/** The AI service returned an assessment that contradicts itself. */
export class InvalidMobilityAssessmentError extends DomainError {}

export interface MobilityAssessmentProps {
  probability: number;
  zone: MobilityZone;
  lowerBound: number;
  upperBound: number;
}

/**
 * The AI's deterministic verdict on a character (selective classification):
 *   probability >= upperBound -> 'mobility'     (the model is confident)
 *   probability <= lowerBound -> 'no_mobility'  (the model is confident)
 *   otherwise                 -> 'uncertain'    (the model abstains; luck decides)
 * The zone is checked against the probability and the band, so an
 * inconsistent answer from the AI service is rejected instead of played.
 */
export class MobilityAssessment {
  readonly probability: number;
  readonly zone: MobilityZone;
  readonly lowerBound: number;
  readonly upperBound: number;

  constructor(props: MobilityAssessmentProps) {
    for (const [name, value] of Object.entries({
      probability: props.probability,
      lowerBound: props.lowerBound,
      upperBound: props.upperBound,
    })) {
      if (typeof value !== 'number' || !Number.isFinite(value) || value < 0 || value > 1) {
        throw new InvalidMobilityAssessmentError(`'${name}' must be a number between 0 and 1, got ${String(value)}`);
      }
    }
    if (!(props.lowerBound < props.upperBound)) {
      throw new InvalidMobilityAssessmentError(
        `lowerBound ${props.lowerBound} must be below upperBound ${props.upperBound}`,
      );
    }
    if (!MOBILITY_ZONES.includes(props.zone)) {
      throw new InvalidMobilityAssessmentError(`'zone' must be one of ${MOBILITY_ZONES.join(', ')}, got ${String(props.zone)}`);
    }
    const expected = MobilityAssessment.zoneOf(props.probability, props.lowerBound, props.upperBound);
    if (props.zone !== expected) {
      throw new InvalidMobilityAssessmentError(
        `zone '${props.zone}' contradicts probability ${props.probability} and band [${props.lowerBound}, ${props.upperBound}]`,
      );
    }
    this.probability = props.probability;
    this.zone = props.zone;
    this.lowerBound = props.lowerBound;
    this.upperBound = props.upperBound;
    Object.freeze(this);
  }

  static zoneOf(probability: number, lowerBound: number, upperBound: number): MobilityZone {
    if (probability >= upperBound) {
      return 'mobility';
    }
    if (probability <= lowerBound) {
      return 'no_mobility';
    }
    return 'uncertain';
  }
}
