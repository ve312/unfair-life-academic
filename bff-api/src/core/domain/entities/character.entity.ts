import { InvalidCharacterError } from '../errors/domain.error.js';

/**
 * Numeric ranges the ML model saw in training. Values outside them would ask
 * the model to extrapolate, so the domain rejects them. They mirror the
 * inference API's domain (ai-service/inference-api/src/domain/entities/character.py).
 */
export const CHARACTER_LIMITS = {
  age: { min: 17, max: 90 },
  educationNum: { min: 1, max: 16 },
  hoursPerWeek: { min: 1, max: 99 },
} as const;

const MAX_CATEGORY_LENGTH = 64;

export interface CharacterProps {
  age: number;
  workclass: string;
  education: string;
  educationNum: number;
  maritalStatus: string;
  occupation: string;
  relationship: string;
  race: string;
  sex: string;
  hoursPerWeek: number;
  nativeCountry: string;
  capitalGain?: number;
  capitalLoss?: number;
  fnlwgt?: number | null;
}

/**
 * The player's character: the 14 traits the mobility model is trained on.
 *
 * Categorical values use the census spellings (e.g. workclass 'Private',
 * education 'HS-grad', nativeCountry 'Mexico'). Values the model never saw
 * are allowed; they simply give no statistical advantage. fnlwgt is a census
 * sampling weight with no in-game meaning: leave it null and the AI service
 * fills it with the training median.
 */
export class Character {
  readonly age: number;
  readonly workclass: string;
  readonly education: string;
  readonly educationNum: number;
  readonly maritalStatus: string;
  readonly occupation: string;
  readonly relationship: string;
  readonly race: string;
  readonly sex: string;
  readonly hoursPerWeek: number;
  readonly nativeCountry: string;
  readonly capitalGain: number;
  readonly capitalLoss: number;
  readonly fnlwgt: number | null;

  constructor(props: CharacterProps) {
    this.age = Character.integerInRange('age', props.age, CHARACTER_LIMITS.age);
    this.workclass = Character.category('workclass', props.workclass);
    this.education = Character.category('education', props.education);
    this.educationNum = Character.integerInRange('educationNum', props.educationNum, CHARACTER_LIMITS.educationNum);
    this.maritalStatus = Character.category('maritalStatus', props.maritalStatus);
    this.occupation = Character.category('occupation', props.occupation);
    this.relationship = Character.category('relationship', props.relationship);
    this.race = Character.category('race', props.race);
    this.sex = Character.category('sex', props.sex);
    this.hoursPerWeek = Character.integerInRange('hoursPerWeek', props.hoursPerWeek, CHARACTER_LIMITS.hoursPerWeek);
    this.nativeCountry = Character.category('nativeCountry', props.nativeCountry);
    this.capitalGain = Character.nonNegativeInteger('capitalGain', props.capitalGain ?? 0);
    this.capitalLoss = Character.nonNegativeInteger('capitalLoss', props.capitalLoss ?? 0);
    this.fnlwgt = props.fnlwgt == null ? null : Character.positiveInteger('fnlwgt', props.fnlwgt);
    Object.freeze(this);
  }

  private static category(name: string, value: unknown): string {
    if (typeof value !== 'string' || value.trim().length === 0) {
      throw new InvalidCharacterError(`'${name}' must be a non-empty string`);
    }
    const trimmed = value.trim();
    if (trimmed.length > MAX_CATEGORY_LENGTH) {
      throw new InvalidCharacterError(`'${name}' must be at most ${MAX_CATEGORY_LENGTH} characters`);
    }
    return trimmed;
  }

  private static integer(name: string, value: unknown): number {
    if (typeof value !== 'number' || !Number.isInteger(value)) {
      throw new InvalidCharacterError(`'${name}' must be an integer, got ${String(value)}`);
    }
    return value;
  }

  private static integerInRange(name: string, value: unknown, range: { min: number; max: number }): number {
    const integer = Character.integer(name, value);
    if (integer < range.min || integer > range.max) {
      throw new InvalidCharacterError(`'${name}' must be between ${range.min} and ${range.max}, got ${integer}`);
    }
    return integer;
  }

  private static nonNegativeInteger(name: string, value: unknown): number {
    const integer = Character.integer(name, value);
    if (integer < 0) {
      throw new InvalidCharacterError(`'${name}' cannot be negative, got ${integer}`);
    }
    return integer;
  }

  private static positiveInteger(name: string, value: unknown): number {
    const integer = Character.integer(name, value);
    if (integer <= 0) {
      throw new InvalidCharacterError(`'${name}' must be positive, got ${integer}`);
    }
    return integer;
  }
}
