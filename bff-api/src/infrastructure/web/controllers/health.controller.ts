import { Controller, Get } from '@nestjs/common';

/**
 * Liveness endpoint for the hosting platform's health check (e.g. Render's
 * "Health Check Path"). It never calls Strapi or the AI service, so it answers
 * immediately and a sleeping dependency cannot make the BFF look unhealthy.
 */
@Controller('health')
export class HealthController {
  @Get()
  check(): { status: 'ok' } {
    return { status: 'ok' };
  }
}
