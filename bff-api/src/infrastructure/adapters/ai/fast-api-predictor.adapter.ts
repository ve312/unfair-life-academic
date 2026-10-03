import { Inject, Injectable, Logger } from '@nestjs/common';
import type { Character } from '../../../core/domain/entities/character.entity.js';
import {
  InvalidMobilityAssessmentError,
  MobilityAssessment,
  type MobilityZone,
} from '../../../core/domain/entities/mobility-assessment.entity.js';
import type { AiPredictorPort } from '../../../core/ports/out/ai-predictor.port.js';
import { AI_SERVICE_CONFIG, type AiServiceConfig } from '../../config/external-services.config.js';
import { ExternalServiceError } from '../external-service.error.js';
import { describeErrorBody, requestJson } from '../http/request-json.js';

const SERVICE = 'ai-service';
const PREDICT_PATH = '/predict';
/** Shared-secret header the AI service checks (ai-service/inference-api security.py). */
export const BFF_TOKEN_HEADER = 'X-BFF-Token';

/** Request body of the FastAPI POST /predict endpoint (ai-service/inference-api schemas.py). */
export interface PredictionRequestPayload {
  age: number;
  workclass: string;
  education: string;
  education_num: number;
  marital_status: string;
  occupation: string;
  relationship: string;
  race: string;
  sex: string;
  hours_per_week: number;
  native_country: string;
  capital_gain: number;
  capital_loss: number;
  fnlwgt?: number;
}

/** Maps the domain Character to the snake_case payload FastAPI validates strictly. */
export function toPredictionPayload(character: Character): PredictionRequestPayload {
  const payload: PredictionRequestPayload = {
    age: character.age,
    workclass: character.workclass,
    education: character.education,
    education_num: character.educationNum,
    marital_status: character.maritalStatus,
    occupation: character.occupation,
    relationship: character.relationship,
    race: character.race,
    sex: character.sex,
    hours_per_week: character.hoursPerWeek,
    native_country: character.nativeCountry,
    capital_gain: character.capitalGain,
    capital_loss: character.capitalLoss,
  };
  // Omitted rather than sent as null: the AI service then fills in its training median.
  if (character.fnlwgt !== null) {
    payload.fnlwgt = character.fnlwgt;
  }
  return payload;
}

/** AiPredictorPort backed by the FastAPI inference service. */
@Injectable()
export class FastApiPredictorAdapter implements AiPredictorPort {
  private readonly logger = new Logger(FastApiPredictorAdapter.name);

  constructor(@Inject(AI_SERVICE_CONFIG) private readonly config: AiServiceConfig) {}

  async assessMobility(character: Character): Promise<MobilityAssessment> {
    const { status, body } = await requestJson(SERVICE, `${this.config.baseUrl}${PREDICT_PATH}`, {
      method: 'POST',
      headers: { [BFF_TOKEN_HEADER]: this.config.token },
      body: toPredictionPayload(character),
      timeoutMs: this.config.timeoutMs,
    });

    if (status === 401 || status === 403) {
      this.logger.error(
        `AI service rejected POST ${PREDICT_PATH} with ${status}; check that BFF_SECRET_TOKEN matches ai-service/inference-api/.env`,
      );
      throw new ExternalServiceError(SERVICE, `POST ${PREDICT_PATH} was rejected (${status}): the BFF token is not accepted`, {
        status,
      });
    }
    if (status !== 200) {
      throw new ExternalServiceError(SERVICE, `POST ${PREDICT_PATH} responded ${status}: ${describeErrorBody(body)}`, {
        status,
      });
    }

    const prediction = body as Record<string, unknown> | null;
    if (
      prediction === null ||
      typeof prediction !== 'object' ||
      typeof prediction.probability !== 'number' ||
      typeof prediction.zone !== 'string' ||
      typeof prediction.lower_bound !== 'number' ||
      typeof prediction.upper_bound !== 'number'
    ) {
      throw new ExternalServiceError(SERVICE, `POST ${PREDICT_PATH} returned an unexpected body: ${describeErrorBody(body)}`, {
        status,
      });
    }

    let assessment: MobilityAssessment;
    try {
      assessment = new MobilityAssessment({
        probability: prediction.probability,
        zone: prediction.zone as MobilityZone,
        lowerBound: prediction.lower_bound,
        upperBound: prediction.upper_bound,
      });
    } catch (error) {
      if (error instanceof InvalidMobilityAssessmentError) {
        throw new ExternalServiceError(SERVICE, `POST ${PREDICT_PATH} returned an inconsistent assessment: ${error.message}`, {
          status,
          cause: error,
        });
      }
      throw error;
    }

    this.logger.debug(
      `Mobility assessment for ${character.nativeCountry}/${character.education}: ` +
        `probability=${assessment.probability.toFixed(4)} zone=${assessment.zone}`,
    );
    return assessment;
  }
}
