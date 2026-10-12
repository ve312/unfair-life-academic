import { IsInt, IsNotEmpty, IsOptional, IsPositive, IsString, Max, MaxLength, Min } from 'class-validator';
import { Character, CHARACTER_LIMITS } from '../../../core/domain/entities/character.entity.js';

const MAX_CATEGORY_LENGTH = 64;

/**
 * The player's character as sent by the game client: the 14 traits of the
 * mobility model, in camelCase. Categorical values use the census spellings
 * (e.g. workclass 'Private', education 'HS-grad', nativeCountry 'Mexico').
 * Numeric limits come from the domain so HTTP validation and domain rules
 * can never disagree.
 */
export class CharacterDto {
  @IsInt()
  @Min(CHARACTER_LIMITS.age.min)
  @Max(CHARACTER_LIMITS.age.max)
  age!: number;

  @IsString()
  @IsNotEmpty()
  @MaxLength(MAX_CATEGORY_LENGTH)
  workclass!: string;

  @IsString()
  @IsNotEmpty()
  @MaxLength(MAX_CATEGORY_LENGTH)
  education!: string;

  @IsInt()
  @Min(CHARACTER_LIMITS.educationNum.min)
  @Max(CHARACTER_LIMITS.educationNum.max)
  educationNum!: number;

  @IsString()
  @IsNotEmpty()
  @MaxLength(MAX_CATEGORY_LENGTH)
  maritalStatus!: string;

  @IsString()
  @IsNotEmpty()
  @MaxLength(MAX_CATEGORY_LENGTH)
  occupation!: string;

  @IsString()
  @IsNotEmpty()
  @MaxLength(MAX_CATEGORY_LENGTH)
  relationship!: string;

  @IsString()
  @IsNotEmpty()
  @MaxLength(MAX_CATEGORY_LENGTH)
  race!: string;

  @IsString()
  @IsNotEmpty()
  @MaxLength(MAX_CATEGORY_LENGTH)
  sex!: string;

  @IsInt()
  @Min(CHARACTER_LIMITS.hoursPerWeek.min)
  @Max(CHARACTER_LIMITS.hoursPerWeek.max)
  hoursPerWeek!: number;

  @IsString()
  @IsNotEmpty()
  @MaxLength(MAX_CATEGORY_LENGTH)
  nativeCountry!: string;

  @IsOptional()
  @IsInt()
  @Min(0)
  capitalGain?: number;

  @IsOptional()
  @IsInt()
  @Min(0)
  capitalLoss?: number;

  @IsOptional()
  @IsInt()
  @IsPositive()
  fnlwgt?: number;

  /** Builds the domain entity; throws InvalidCharacterError if a domain rule is broken. */
  toDomain(): Character {
    return new Character({
      age: this.age,
      workclass: this.workclass,
      education: this.education,
      educationNum: this.educationNum,
      maritalStatus: this.maritalStatus,
      occupation: this.occupation,
      relationship: this.relationship,
      race: this.race,
      sex: this.sex,
      hoursPerWeek: this.hoursPerWeek,
      nativeCountry: this.nativeCountry,
      capitalGain: this.capitalGain,
      capitalLoss: this.capitalLoss,
      fnlwgt: this.fnlwgt ?? null,
    });
  }
}
