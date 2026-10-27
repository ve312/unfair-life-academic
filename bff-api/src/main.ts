import { Logger } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import { NestFactory } from '@nestjs/core';
import { AppModule } from './app.module.js';
import { createCorsOriginCheck, loadHttpConfig } from './infrastructure/config/http.config.js';
import { createValidationPipe } from './infrastructure/web/validation-pipe.js';

async function bootstrap(): Promise<void> {
  const app = await NestFactory.create(AppModule);
  const config = app.get(ConfigService);
  const http = loadHttpConfig({
    NODE_ENV: config.get<string>('NODE_ENV'),
    PORT: config.get<string>('PORT'),
    FRONTEND_URL: config.get<string>('FRONTEND_URL'),
    CORS_ORIGINS: config.get<string>('CORS_ORIGINS'),
  });

  app.enableCors({
    origin: createCorsOriginCheck(http.corsOrigins, (origin) =>
      Logger.warn(`CORS rejected origin ${origin}; add it to FRONTEND_URL or CORS_ORIGINS`, 'Bootstrap'),
    ),
    methods: ['GET', 'POST', 'OPTIONS'],
    allowedHeaders: ['Content-Type', 'Accept'],
    maxAge: 600,
  });
  app.useGlobalPipes(createValidationPipe());
  app.enableShutdownHooks();

  // PORT comes from the platform (Render) or defaults to 3000; 0.0.0.0 so it is reachable from outside the container.
  await app.listen(http.port, http.host);
  Logger.log(`BFF listening on ${http.host}:${http.port}; CORS allowed for ${http.corsOrigins.join(', ')}`, 'Bootstrap');
}

await bootstrap();
