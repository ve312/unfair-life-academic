import { Type } from 'class-transformer';
import { IsNotEmpty, IsObject, IsOptional, IsString, MaxLength, ValidateNested } from 'class-validator';
import { CharacterDto } from './character.dto.js';

/** Strapi documentIds are 24 characters; the margin allows other id schemes. */
const MAX_ID_LENGTH = 64;

/** Body of POST /game/decision: the player's choice in the current scene. */
export class SubmitChoiceDto {
  @IsString()
  @IsNotEmpty()
  @MaxLength(MAX_ID_LENGTH)
  currentNodeId!: string;

  @IsString()
  @IsNotEmpty()
  @MaxLength(MAX_ID_LENGTH)
  decisionId!: string;

  /**
   * @deprecated Ignored. Mobility checks evaluate the story's base_profile, defined in the CMS.
   * Still accepted (and validated) so that older game clients that send it are not rejected.
   */
  @IsOptional()
  @IsObject()
  @ValidateNested()
  @Type(() => CharacterDto)
  character?: CharacterDto;
}
