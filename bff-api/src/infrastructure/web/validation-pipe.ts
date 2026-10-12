import { ValidationPipe } from '@nestjs/common';

/**
 * Request validation shared by the server (main.ts) and the e2e tests:
 * unknown properties are rejected, and types stay strict, so "34" is not
 * accepted where an integer is required.
 */
export const createValidationPipe = (): ValidationPipe =>
  new ValidationPipe({
    whitelist: true,
    forbidNonWhitelisted: true,
    transform: true,
    transformOptions: { enableImplicitConversion: false },
  });
