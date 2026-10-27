import type { Core } from '@strapi/strapi';

const allowedMediaTypes = [
  'image/*',
  'video/*',
  'audio/*',
  'application/pdf',
  'application/msword',
  'application/vnd.openxmlformats-officedocument.*',
  'text/plain',
  'text/csv',
];

const deniedTypes = [
  'image/svg+xml',
  'application/vnd.microsoft.portable-executable',
  'application/x-msdownload',
  'application/x-msdos-program',
  'application/x-executable',
  'application/x-dosexec',
  'application/x-sh',
  'text/x-shellscript',
  'application/x-mach-binary',
];

const CLOUDINARY_VARIABLES = ['CLOUDINARY_NAME', 'CLOUDINARY_KEY', 'CLOUDINARY_SECRET'] as const;

/**
 * Media storage:
 *   - production: Cloudinary (Render's filesystem is ephemeral, so files in
 *     public/uploads would vanish on every deploy and restart).
 *   - development: the default local provider (public/uploads).
 */
function cloudinaryProvider(env: Core.Config.Shared.ConfigParams['env']) {
  const missing = CLOUDINARY_VARIABLES.filter((name) => env(name, '').trim() === '');
  if (missing.length > 0) {
    throw new Error(
      `Cloudinary uploads are required in production; missing environment variables: ${missing.join(', ')}`
    );
  }
  return {
    provider: 'cloudinary',
    providerOptions: {
      cloud_name: env('CLOUDINARY_NAME', '').trim(),
      api_key: env('CLOUDINARY_KEY', '').trim(),
      api_secret: env('CLOUDINARY_SECRET', '').trim(),
    },
    actionOptions: {
      upload: {},
      uploadStream: {},
      delete: {},
    },
  };
}

const config = ({ env }: Core.Config.Shared.ConfigParams): Core.Config.Plugin => ({
  'users-permissions': {
    config: {
      jwtManagement: 'refresh',
      sessions: {
        httpOnly: true,
      },
    },
  },
  upload: {
    config: {
      ...(env('NODE_ENV') === 'production' ? cloudinaryProvider(env) : {}),
      security: {
        allowedTypes: allowedMediaTypes,
        deniedTypes,
      },
    },
  },
});

export default config;
