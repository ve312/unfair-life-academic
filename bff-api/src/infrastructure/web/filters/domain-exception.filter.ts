import { type ArgumentsHost, Catch, type ExceptionFilter, HttpStatus, Logger } from '@nestjs/common';
import type { Response } from 'express';
import {
  BaseProfileRequiredError,
  DecisionNotFoundError,
  DomainError,
  InvalidCharacterError,
  InvalidStoryContentError,
  StoryNodeNotFoundError,
  StoryNotFoundError,
} from '../../../core/domain/errors/domain.error.js';
import { ExternalServiceError } from '../../adapters/external-service.error.js';

interface ErrorMapping {
  status: HttpStatus;
  error: string;
  /** Whether the original message is safe and useful to show the client. */
  exposeMessage: boolean;
}

/**
 * Translates domain and external-service errors into HTTP responses:
 *
 *   StoryNotFoundError, StoryNodeNotFoundError -> 404 Not Found
 *   DecisionNotFoundError                      -> 400 Bad Request (choice not offered in that scene)
 *   BaseProfileRequiredError, InvalidCharacterError -> 422 Unprocessable Entity (message exposed: it says what to fix)
 *   InvalidStoryContentError                   -> 502 Bad Gateway (the CMS holds broken content)
 *   ExternalServiceError (no answer/timeout)   -> 503 Service Unavailable
 *   ExternalServiceError (bad answer)          -> 502 Bad Gateway
 *
 * Upstream details (URLs, statuses) are logged but never sent to the client.
 */
@Catch(DomainError, ExternalServiceError)
export class DomainExceptionFilter implements ExceptionFilter {
  private readonly logger = new Logger(DomainExceptionFilter.name);

  catch(exception: DomainError | ExternalServiceError, host: ArgumentsHost): void {
    const response = host.switchToHttp().getResponse<Response>();
    const mapping = DomainExceptionFilter.map(exception);

    if (mapping.status >= 500) {
      this.logger.error(exception.message);
    } else {
      this.logger.warn(`${mapping.status} ${exception.name}: ${exception.message}`);
    }

    response.status(mapping.status).json({
      statusCode: mapping.status,
      error: mapping.error,
      message: mapping.exposeMessage ? exception.message : DomainExceptionFilter.genericMessage(mapping.status),
    });
  }

  private static map(exception: DomainError | ExternalServiceError): ErrorMapping {
    if (exception instanceof StoryNotFoundError || exception instanceof StoryNodeNotFoundError) {
      return { status: HttpStatus.NOT_FOUND, error: 'Not Found', exposeMessage: true };
    }
    if (exception instanceof DecisionNotFoundError) {
      return { status: HttpStatus.BAD_REQUEST, error: 'Bad Request', exposeMessage: true };
    }
    if (exception instanceof BaseProfileRequiredError || exception instanceof InvalidCharacterError) {
      return { status: HttpStatus.UNPROCESSABLE_ENTITY, error: 'Unprocessable Entity', exposeMessage: true };
    }
    if (exception instanceof InvalidStoryContentError) {
      return { status: HttpStatus.BAD_GATEWAY, error: 'Bad Gateway', exposeMessage: false };
    }
    if (exception instanceof ExternalServiceError) {
      return exception.status === undefined
        ? { status: HttpStatus.SERVICE_UNAVAILABLE, error: 'Service Unavailable', exposeMessage: false }
        : { status: HttpStatus.BAD_GATEWAY, error: 'Bad Gateway', exposeMessage: false };
    }
    return { status: HttpStatus.INTERNAL_SERVER_ERROR, error: 'Internal Server Error', exposeMessage: false };
  }

  private static genericMessage(status: HttpStatus): string {
    switch (status) {
      case HttpStatus.SERVICE_UNAVAILABLE:
        return 'A game service is temporarily unavailable. Please try again.';
      case HttpStatus.BAD_GATEWAY:
        return 'A game service returned an invalid response.';
      default:
        return 'An unexpected error occurred.';
    }
  }
}
