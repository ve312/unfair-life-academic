import { Module } from '@nestjs/common';
import { ConfigModule, ConfigService } from '@nestjs/config';
import { LIST_STORIES_USE_CASE } from './core/ports/in/list-stories.use-case.js';
import { RESOLVE_DECISION_USE_CASE } from './core/ports/in/resolve-decision.use-case.js';
import { START_STORY_USE_CASE } from './core/ports/in/start-story.use-case.js';
import { AI_PREDICTOR_PORT } from './core/ports/out/ai-predictor.port.js';
import { RANDOM_SOURCE } from './core/ports/out/random-source.port.js';
import { STORY_CONTENT_PORT } from './core/ports/out/story-content.port.js';
import { ListStoriesService } from './core/use-cases/list-stories.service.js';
import { ResolveDecisionService } from './core/use-cases/resolve-decision.service.js';
import { StartStoryService } from './core/use-cases/start-story.service.js';
import { FastApiPredictorAdapter } from './infrastructure/adapters/ai/fast-api-predictor.adapter.js';
import { StrapiContentAdapter } from './infrastructure/adapters/cms/strapi-content.adapter.js';
import { CryptoRandomSource } from './infrastructure/adapters/random/crypto-random-source.adapter.js';
import {
  AI_SERVICE_CONFIG,
  loadAiServiceConfig,
  loadStrapiConfig,
  STRAPI_CONFIG,
} from './infrastructure/config/external-services.config.js';
import { GameController } from './infrastructure/web/controllers/game.controller.js';
import { HealthController } from './infrastructure/web/controllers/health.controller.js';

const ENV_KEYS = [
  'NODE_ENV',
  'AI_SERVICE_URL',
  'AI_SERVICE_TIMEOUT_MS',
  'STRAPI_URL',
  'STRAPI_MEDIA_BASE_URL',
  'STRAPI_TIMEOUT_MS',
  'BFF_SECRET_TOKEN',
] as const;

/** Reads the external-service variables through ConfigService (.env file + real environment). */
const envFrom = (config: ConfigService): Record<string, string | undefined> =>
  Object.fromEntries(ENV_KEYS.map((key) => [key, config.get<string>(key)]));

@Module({
  imports: [
    // Loads .env; real environment variables take precedence over the file.
    ConfigModule.forRoot({ isGlobal: true, cache: true }),
  ],
  controllers: [GameController, HealthController],
  providers: [
    // Configuration, validated once at startup.
    { provide: AI_SERVICE_CONFIG, inject: [ConfigService], useFactory: (config: ConfigService) => loadAiServiceConfig(envFrom(config)) },
    { provide: STRAPI_CONFIG, inject: [ConfigService], useFactory: (config: ConfigService) => loadStrapiConfig(envFrom(config)) },

    // Outbound ports -> infrastructure adapters.
    { provide: STORY_CONTENT_PORT, useClass: StrapiContentAdapter },
    { provide: AI_PREDICTOR_PORT, useClass: FastApiPredictorAdapter },
    { provide: RANDOM_SOURCE, useClass: CryptoRandomSource },

    // Inbound ports -> application use cases.
    { provide: START_STORY_USE_CASE, useClass: StartStoryService },
    { provide: RESOLVE_DECISION_USE_CASE, useClass: ResolveDecisionService },
    { provide: LIST_STORIES_USE_CASE, useClass: ListStoriesService },
  ],
})
export class AppModule {}
