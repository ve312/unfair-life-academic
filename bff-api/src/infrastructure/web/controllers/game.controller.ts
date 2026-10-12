import { Body, Controller, Get, HttpCode, HttpStatus, Inject, Param, Post, UseFilters } from '@nestjs/common';
import { LIST_STORIES_USE_CASE, type ListStoriesUseCase } from '../../../core/ports/in/list-stories.use-case.js';
import { RESOLVE_DECISION_USE_CASE, type ResolveDecisionUseCase } from '../../../core/ports/in/resolve-decision.use-case.js';
import { START_STORY_USE_CASE, type StartStoryUseCase } from '../../../core/ports/in/start-story.use-case.js';
import { type StorySummaryDto, toStorySummaryDto } from '../dtos/story-summary.dto.js';
import { SubmitChoiceDto } from '../dtos/submit-choice.dto.js';
import { DomainExceptionFilter } from '../filters/domain-exception.filter.js';
import { presentStoryNode, type StoryNodeView } from '../presenters/story-node.presenter.js';

/** Game flow endpoints used by the Phaser client. */
@Controller('game')
@UseFilters(DomainExceptionFilter)
export class GameController {
  constructor(
    @Inject(START_STORY_USE_CASE) private readonly startStory: StartStoryUseCase,
    @Inject(RESOLVE_DECISION_USE_CASE) private readonly resolveDecision: ResolveDecisionUseCase,
    @Inject(LIST_STORIES_USE_CASE) private readonly listStories: ListStoriesUseCase,
  ) {}

  /** Lists the stories the player can start, for the story-selection menu. */
  @Get('stories')
  async stories(): Promise<StorySummaryDto[]> {
    return (await this.listStories.execute()).map(toStorySummaryDto);
  }

  /** Starts a story and returns its first scene. */
  @Post('start/:storyId')
  @HttpCode(HttpStatus.OK)
  async start(@Param('storyId') storyId: string): Promise<StoryNodeView> {
    return presentStoryNode(await this.startStory.execute(storyId));
  }

  /** Applies the player's choice and returns the scene it leads to. */
  @Post('decision')
  @HttpCode(HttpStatus.OK)
  async decide(@Body() choice: SubmitChoiceDto): Promise<StoryNodeView> {
    // choice.character is accepted for older clients but ignored: mobility checks use the story's base profile.
    return presentStoryNode(await this.resolveDecision.execute(choice.currentNodeId, choice.decisionId));
  }
}
