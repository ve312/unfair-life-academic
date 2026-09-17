import path from 'path';
import type { Core } from '@strapi/strapi';
import { isDatabaseClientKind } from '@strapi/database';

/**
 * Database selection:
 *   - production (NODE_ENV=production): PostgreSQL through DATABASE_URL (Supabase).
 *     SQLite is refused: Render's filesystem is wiped on every deploy and restart.
 *   - development: SQLite in .tmp/data.db (DATABASE_FILENAME to change it).
 * DATABASE_CLIENT still overrides the choice, e.g. to try PostgreSQL locally.
 */
const config = ({ env }: Core.Config.Shared.ConfigParams): Core.Config.Database => {
  const isProduction = env('NODE_ENV') === 'production';
  const client = env('DATABASE_CLIENT', isProduction ? 'postgres' : 'sqlite');

  if (!isDatabaseClientKind(client)) {
    throw new Error(
      `Unsupported DATABASE_CLIENT: ${client}. Use "postgres", "mysql", or "sqlite".`
    );
  }
  if (isProduction && client === 'sqlite') {
    throw new Error(
      'SQLite is not allowed in production: the hosting filesystem is ephemeral and the data would be lost. ' +
        'Set DATABASE_URL to the PostgreSQL connection string (and leave DATABASE_CLIENT unset or "postgres").'
    );
  }

  const databaseUrl = env('DATABASE_URL', '').trim();
  if (isProduction && client === 'postgres' && databaseUrl === '') {
    throw new Error('DATABASE_URL must be set in production (the Supabase PostgreSQL connection string).');
  }

  // Supabase requires TLS. Its certificate is signed by Supabase's own CA: paste that CA
  // (Project Settings > Database > SSL) into DATABASE_SSL_CA to verify the server.
  // Without it the connection is still encrypted, but the certificate is not verified.
  const sslCa = env('DATABASE_SSL_CA', '').replace(/\\n/g, '\n').trim() || undefined;
  const postgresSsl = env.bool('DATABASE_SSL', isProduction) && {
    ca: sslCa,
    rejectUnauthorized: env.bool('DATABASE_SSL_REJECT_UNAUTHORIZED', sslCa !== undefined),
  };

  const connections: Record<Core.Config.Database.ClientKind, Core.Config.Database['connection']> = {
    mysql: {
      client: 'mysql',
      connection: {
        host: env('DATABASE_HOST', 'localhost'),
        port: env.int('DATABASE_PORT', 3306),
        database: env('DATABASE_NAME', 'strapi'),
        user: env('DATABASE_USERNAME', 'strapi'),
        password: env('DATABASE_PASSWORD', 'strapi'),
        ssl: env.bool('DATABASE_SSL', false) && {
          key: env('DATABASE_SSL_KEY', undefined),
          cert: env('DATABASE_SSL_CERT', undefined),
          ca: env('DATABASE_SSL_CA', undefined),
          capath: env('DATABASE_SSL_CAPATH', undefined),
          cipher: env('DATABASE_SSL_CIPHER', undefined),
          rejectUnauthorized: env.bool('DATABASE_SSL_REJECT_UNAUTHORIZED', true),
        },
      },
      pool: { min: env.int('DATABASE_POOL_MIN', 2), max: env.int('DATABASE_POOL_MAX', 10) },
    },
    postgres: {
      client: 'postgres',
      connection: {
        // pg uses connectionString when set; the separate fields serve a local PostgreSQL.
        connectionString: databaseUrl || undefined,
        host: env('DATABASE_HOST', 'localhost'),
        port: env.int('DATABASE_PORT', 5432),
        database: env('DATABASE_NAME', 'strapi'),
        user: env('DATABASE_USERNAME', 'strapi'),
        password: env('DATABASE_PASSWORD', 'strapi'),
        ssl: postgresSsl,
        schema: env('DATABASE_SCHEMA', 'public'),
      },
      // Free-tier poolers allow few connections, and an idle free instance should hold none.
      pool: {
        min: env.int('DATABASE_POOL_MIN', isProduction ? 0 : 2),
        max: env.int('DATABASE_POOL_MAX', isProduction ? 5 : 10),
      },
    },
    sqlite: {
      client: 'sqlite',
      connection: {
        // `||` rather than env()'s default: an empty DATABASE_FILENAME= line would otherwise
        // resolve to the project directory itself and SQLite fails with "unable to open database file".
        filename: path.join(__dirname, '..', '..', env('DATABASE_FILENAME', '').trim() || '.tmp/data.db'),
      },
      useNullAsDefault: true,
    },
  };

  return {
    connection: {
      ...connections[client],
      acquireConnectionTimeout: env.int('DATABASE_CONNECTION_TIMEOUT', 60000),
    },
  };
};

export default config;
