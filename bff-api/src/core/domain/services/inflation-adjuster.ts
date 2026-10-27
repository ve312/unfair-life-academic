import { Character } from '../entities/character.entity.js';
import { MODEL_BASE_YEAR, toModelBaseYearDollars } from './consumer-price-index.js';

/**
 * Returns the character with its money-valued traits converted from `year`
 * dollars to 1994 dollars, the only currency the mobility model understands.
 * capital_gain and capital_loss are the model's only monetary inputs (income is
 * what it predicts, not an input), so every other trait is copied unchanged.
 * Amounts are rounded to whole dollars, as in the census data.
 */
export function deflateToModelBaseYear(character: Character, year: number): Character {
  if (year === MODEL_BASE_YEAR) {
    return character;
  }
  return new Character({
    age: character.age,
    workclass: character.workclass,
    education: character.education,
    educationNum: character.educationNum,
    maritalStatus: character.maritalStatus,
    occupation: character.occupation,
    relationship: character.relationship,
    race: character.race,
    sex: character.sex,
    hoursPerWeek: character.hoursPerWeek,
    nativeCountry: character.nativeCountry,
    capitalGain: Math.round(toModelBaseYearDollars(character.capitalGain, year)),
    capitalLoss: Math.round(toModelBaseYearDollars(character.capitalLoss, year)),
    fnlwgt: character.fnlwgt,
  });
}
