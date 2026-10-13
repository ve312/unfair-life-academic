/**
 * One-off data migration: give the stories created before `base_profile` existed a
 * sociodemographic profile consistent with their narrative, and link their orphan scenes.
 *
 * Usage, from the cms folder (it uses the same database settings as Strapi, so it
 * migrates the local SQLite database or, with the production environment, PostgreSQL):
 *
 *     node scripts/migrate-base-profiles.js --dry-run   # report only
 *     node scripts/migrate-base-profiles.js             # apply
 *
 * What it does, and why:
 *
 * 1. Profiles. Each known story (matched by its unique title) receives the profile
 *    below, but only when it has none: a profile an editor already set is never
 *    overwritten. Writes go through the document service, so the story lifecycle
 *    validates every profile exactly as the admin panel would.
 * 2. Orphan scenes. The BFF finds a scene's base profile (and year) through the
 *    scene -> story relation. Scenes reachable from a story's starting scene that are
 *    linked to no story are linked to it; without this, a mobility check in such a
 *    scene would fail even after the story receives a profile. A scene already linked
 *    to another story (a scene shared between stories) is never reassigned.
 *
 * The script is idempotent: running it again changes nothing.
 *
 * Profile design. Each profile encodes who the protagonist is when the story begins,
 * in the census vocabulary of the model. Scores of the deployed model (band [0.23, 0.78]):
 *
 * - El Precio del Ascenso: a 20-year-old mechanic who wakes under a zinc roof, works
 *   14 hours a day (84 h/week) and supports a hungry younger brother. p = 0.0015, a
 *   deterministic "no": the story asks whether merit is enough, and the census answers.
 * - La Prueba Definitiva: the demo of selective classification. A 31-year-old married
 *   technician with a degree who goes hungry and risks his last savings on the trip to
 *   the interview. p = 0.33 falls inside the uncertainty band, so the weighted coin
 *   ("systemic luck") decides, which is what the demo exists to show.
 * - Presentación: Juan Sánchez, from La Guajira (stratum 1), who left school at 10 to
 *   pick coffee and is now a 20-year-old single father. p = 0.0010. The story has no
 *   mobility check today; the profile makes it ready for one.
 * - prueba 1: a test story whose first branch parodies Breaking Bad: a 50-year-old
 *   public-school chemistry teacher. p = 0.75. No mobility check today.
 *
 * "Columbia" (not "Colombia") is the census spelling the model was trained on.
 */
const { createStrapi, compileStrapi } = require('@strapi/strapi');

const PROFILES = {
  'El Precio del Ascenso': {
    age: 20, workclass: 'Private', education: 'HS-grad', 'education-num': 9, 'marital-status': 'Never-married',
    occupation: 'Craft-repair', relationship: 'Own-child', race: 'Other', sex: 'Male', 'hours-per-week': 84,
    'native-country': 'Columbia', 'capital-gain': 0, 'capital-loss': 0,
  },
  'La Prueba Definitiva': {
    age: 31, workclass: 'Private', education: 'Bachelors', 'education-num': 13, 'marital-status': 'Married-civ-spouse',
    occupation: 'Tech-support', relationship: 'Husband', race: 'Other', sex: 'Male', 'hours-per-week': 45,
    'native-country': 'Columbia', 'capital-gain': 0, 'capital-loss': 0,
  },
  Presentación: {
    age: 20, workclass: 'Private', education: '1st-4th', 'education-num': 2, 'marital-status': 'Never-married',
    occupation: 'Farming-fishing', relationship: 'Unmarried', race: 'Other', sex: 'Male', 'hours-per-week': 60,
    'native-country': 'Columbia', 'capital-gain': 0, 'capital-loss': 0,
  },
  'prueba 1': {
    age: 50, workclass: 'Local-gov', education: 'Masters', 'education-num': 14, 'marital-status': 'Married-civ-spouse',
    occupation: 'Prof-specialty', relationship: 'Husband', race: 'White', sex: 'Male', 'hours-per-week': 50,
    'native-country': 'United-States', 'capital-gain': 0, 'capital-loss': 0,
  },
};

const isEmpty = (value) => value === null || value === undefined
  || (typeof value === 'object' && !Array.isArray(value) && Object.keys(value).length === 0);

/** Scenes reachable from the starting scene through decisions (next and failure scenes). */
async function reachableScenes(app, startingSceneId) {
  const scenes = [];
  const seen = new Set();
  const queue = [startingSceneId];
  while (queue.length > 0) {
    const documentId = queue.shift();
    if (seen.has(documentId)) continue;
    seen.add(documentId);
    const scene = await app.documents('api::scene.scene').findOne({
      documentId,
      fields: ['title'],
      populate: {
        story: { fields: ['title'] },
        available_decisions: { fields: ['text'], populate: { next_scene: { fields: ['title'] }, failure_scene: { fields: ['title'] } } },
      },
    });
    if (!scene) continue;
    scenes.push(scene);
    for (const decision of scene.available_decisions ?? []) {
      for (const target of [decision.next_scene, decision.failure_scene]) {
        if (target) queue.push(target.documentId);
      }
    }
  }
  return scenes;
}

async function main() {
  const dryRun = process.argv.includes('--dry-run');
  const app = await createStrapi(await compileStrapi()).load();
  app.log.level = 'error';
  const report = [];
  try {
    const stories = await app.documents('api::story.story').findMany({
      fields: ['title', 'base_profile'],
      populate: { starting_scene: { fields: ['title'] } },
    });
    for (const story of stories) {
      const profile = PROFILES[story.title];
      if (profile === undefined) {
        report.push(`skip   "${story.title}": no profile designed for this story`);
        continue;
      }
      if (isEmpty(story.base_profile)) {
        if (!dryRun) {
          await app.documents('api::story.story').update({ documentId: story.documentId, data: { base_profile: profile } });
        }
        report.push(`${dryRun ? 'would set' : 'set'}    "${story.title}": base_profile`);
      } else {
        report.push(`keep   "${story.title}": already has a base_profile`);
      }

      if (!story.starting_scene) continue;
      const scenes = await reachableScenes(app, story.starting_scene.documentId);
      const orphans = scenes.filter((scene) => !scene.story);
      const shared = scenes.filter((scene) => scene.story && scene.story.documentId !== story.documentId);
      for (const scene of orphans) {
        if (!dryRun) {
          await app.documents('api::scene.scene').update({ documentId: scene.documentId, data: { story: story.documentId } });
        }
      }
      if (orphans.length > 0) {
        report.push(`${dryRun ? 'would link' : 'link'}   "${story.title}": ${orphans.length} orphan scene(s) of ${scenes.length} reachable`);
      }
      for (const scene of shared) {
        report.push(`note   "${story.title}": scene "${scene.title.slice(0, 60)}" belongs to "${scene.story.title}" (left unchanged)`);
      }
    }
  } finally {
    console.log(report.join('\n'));
    await app.destroy();
  }
  process.exit(0);
}

main().catch((error) => {
  console.error(`Migration failed: ${error.name}: ${error.message}`);
  process.exit(1);
});
