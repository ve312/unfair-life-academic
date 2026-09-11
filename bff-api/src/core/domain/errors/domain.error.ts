/**
 * Errors raised by the domain layer. Infrastructure maps them to transport
 * errors (e.g. HTTP 422) without the domain knowing about HTTP.
 */
export class DomainError extends Error {
  constructor(message: string) {
    super(message);
    this.name = new.target.name;
  }
}

/** A character profile violates a domain rule (missing trait, out-of-range value). */
export class InvalidCharacterError extends DomainError {}

/** Story content (a node or a decision) is malformed or breaks a graph rule. */
export class InvalidStoryContentError extends DomainError {}

/** A requested story node does not exist in the content source. */
export class StoryNodeNotFoundError extends DomainError {
  constructor(public readonly nodeId: string) {
    super(`Story node '${nodeId}' was not found`);
  }
}

/** A requested story does not exist in the content source. */
export class StoryNotFoundError extends DomainError {
  constructor(public readonly storyId: string) {
    super(`Story '${storyId}' was not found`);
  }
}

/** The player chose a decision that is not offered in the current scene. */
export class DecisionNotFoundError extends DomainError {
  constructor(
    public readonly nodeId: string,
    public readonly decisionId: string,
  ) {
    super(`Decision '${decisionId}' is not available in story node '${nodeId}'`);
  }
}

/**
 * A decision triggers the AI (mobility check) but the story it belongs to declares no base_profile.
 * Profiles are optional because purely narrative stories never consult the AI; a story that does must define one.
 */
export class BaseProfileRequiredError extends DomainError {
  constructor(
    public readonly decisionId: string,
    public readonly nodeId: string,
  ) {
    super(
      `Decision '${decisionId}' in scene '${nodeId}' requires the AI, but its story has no base_profile defined. ` +
        'Define the base profile in the CMS, or link the scene to its story.',
    );
  }
}
