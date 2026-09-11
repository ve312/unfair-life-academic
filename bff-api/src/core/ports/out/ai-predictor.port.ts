import type { Character } from '../../domain/entities/character.entity.js';
import type { MobilityAssessment } from '../../domain/entities/mobility-assessment.entity.js';

/** Injection token for AiPredictorPort (a runtime value, since interfaces vanish at compile time). */
export const AI_PREDICTOR_PORT = Symbol('AiPredictorPort');

/**
 * Outbound port: the AI service that assesses a character's social mobility.
 *
 * The character's money-valued traits must already be in 1994 dollars (the
 * model's training year). The assessment is deterministic; turning an
 * 'uncertain' assessment into an outcome is a game rule applied by the BFF.
 * Implementations must not mutate the character.
 */
export interface AiPredictorPort {
  assessMobility(character: Character): Promise<MobilityAssessment>;
}
