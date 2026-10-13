/**
 * Seed: "Turno de Noche", an immersive story about precarious gig work, health-system barriers,
 * predatory lending, the care burden on women and forced migration (Medellín, 2026-2033).
 *
 * Runs from bootstrap (src/index.ts) and is idempotent: it does nothing when a story with this
 * title already exists. Creation is all-or-nothing: if any step fails, every document it created
 * is deleted, so a half-built story can never block a later, successful run.
 *
 * Media were uploaded beforehand to the media library (public-domain / CC0 files from Wikimedia
 * Commons; the source and licence of each one are stored in its caption). They are looked up by
 * their unique file name, not by numeric id, so that in an environment without them (e.g. a local
 * database) the story is created without media instead of being linked to unrelated files.
 *
 * How the AI is used. Daniela (base_profile) is a 24-year-old delivery rider living with her
 * partner; she is a woman with some college. Each mobility check applies hidden_modifiers that
 * describe the life its path opens seven years later. The deployed model (band [0.23, 0.78])
 * then scores that life:
 *
 *   Andrés stays            nurse 0.435 · call-centre supervisor 0.373  -> weighted coin (luck)
 *   Andrés migrates alone   nurse 0.082 · live-in domestic worker 0.011 -> deterministic "no"
 *   the three cross Darién  nursing technician 0.256 -> coin · night cleaning 0.063 -> "no"
 *
 * The contrast is the story's argument. When the household holds together, effort moves the odds
 * and luck decides. When migration breaks it, no decision of hers is enough: the 1994 census the
 * model learned from barely lets a woman who cares alone cross the threshold.
 */
import type { Core } from '@strapi/strapi';

export const STORY_TITLE = 'Turno de Noche';

/** Files uploaded to the media library for this story (unique names). */
const MEDIA = {
  rainStreet: 'turno-de-noche-img-rain-street.jpg',
  motoLight: 'turno-de-noche-img-moto-light.jpg',
  cityTraffic: 'turno-de-noche-img-city-traffic.jpg',
  bogotaNight: 'turno-de-noche-img-bogota-night.jpg',
  medellinDawn: 'turno-de-noche-img-medellin-dawn.jpg',
  erWheelchairs: 'turno-de-noche-img-er-wheelchairs.jpg',
  erEntrance: 'turno-de-noche-img-er-entrance.jpg',
  insulin: 'turno-de-noche-img-insulin.jpg',
  debtWall: 'turno-de-noche-img-debt-wall.jpg',
  timeMoney: 'turno-de-noche-img-time-money.jpg',
  phoneNight: 'turno-de-noche-img-phone-night.jpg',
  rainWindow: 'turno-de-noche-img-rain-window.jpg',
  jungleDark: 'turno-de-noche-img-jungle-dark.jpg',
  houstonNight: 'turno-de-noche-img-houston-night.jpg',
  graduation: 'turno-de-noche-img-graduation.jpg',
  planeSunset: 'turno-de-noche-img-plane-sunset.jpg',
  callCenter: 'turno-de-noche-img-call-center.jpg',
  kitchen: 'turno-de-noche-img-kitchen.jpg',
  cleanOffice: 'turno-de-noche-img-clean-office.jpg',
  womanAhead: 'turno-de-noche-img-woman-ahead.jpg',
  forestLight: 'turno-de-noche-img-forest-light.jpg',
  riverVideo: 'turno-de-noche-vid-river.webm',
  audRain: 'turno-de-noche-aud-rain.mp3',
  audStreet: 'turno-de-noche-aud-street.mp3',
  audTension: 'turno-de-noche-aud-tension.mp3',
  audRainforest: 'turno-de-noche-aud-rainforest.mp3',
  audPiano: 'turno-de-noche-aud-piano.mp3',
  audHope: 'turno-de-noche-aud-hope.mp3',
} as const;
type MediaKey = keyof typeof MEDIA;

/** Looping ambience and music, shared by several scenes. */
const AUDIO_TRACKS = {
  rain: { name: 'Turno de Noche · Lluvia', file: 'audRain', volume: 0.55 },
  street: { name: 'Turno de Noche · Calle', file: 'audStreet', volume: 0.45 },
  tension: { name: 'Turno de Noche · Urgencias', file: 'audTension', volume: 0.5 },
  rainforest: { name: 'Turno de Noche · Selva', file: 'audRainforest', volume: 0.6 },
  piano: { name: 'Turno de Noche · Piano', file: 'audPiano', volume: 0.45 },
  hope: { name: 'Turno de Noche · Esperanza', file: 'audHope', volume: 0.45 },
} as const satisfies Record<string, { name: string; file: MediaKey; volume: number }>;
type TrackKey = keyof typeof AUDIO_TRACKS;

/** Daniela at the start of the story, in the census vocabulary of the model. */
const BASE_PROFILE = {
  age: 24,
  workclass: 'Self-emp-not-inc', // a delivery app's "independent partner": no contract, no benefits
  education: 'Some-college', // left nursing in her fourth semester when Sofía was born
  'education-num': 10,
  'marital-status': 'Married-civ-spouse', // common-law union with Andrés
  occupation: 'Transport-moving',
  relationship: 'Wife',
  race: 'Other',
  sex: 'Female',
  'hours-per-week': 72,
  'native-country': 'Columbia', // census spelling
  'capital-gain': 0,
  'capital-loss': 0,
};

/** The life each AI path opens seven years later (2033), applied on top of the base profile. */
const NURSE = { age: 7, workclass: 'Private', occupation: 'Prof-specialty', education: 'Bachelors', 'education-num': 3, 'hours-per-week': -24 };
const ALONE = { 'marital-status': 'Married-spouse-absent', relationship: 'Unmarried' };
const MODIFIERS = {
  nurseTogether: NURSE, // p = 0.435
  supervisorTogether: { age: 7, workclass: 'Private', occupation: 'Exec-managerial', 'hours-per-week': -24 }, // p = 0.373
  nurseAlone: { ...NURSE, ...ALONE }, // p = 0.082
  liveInAlone: { age: 7, workclass: 'Private', occupation: 'Priv-house-serv', 'hours-per-week': -6, ...ALONE }, // p = 0.011
  technicianHouston: { age: 7, workclass: 'Private', occupation: 'Tech-support', education: 'Assoc-voc', 'education-num': 1, 'hours-per-week': -24 }, // p = 0.256
  cleanerHouston: { age: 7, workclass: 'Private', occupation: 'Other-service', 'hours-per-week': -32 }, // p = 0.063
};

interface DecisionSpec {
  text: string;
  next: string;
  /** Mobility check: the scene reached when the AI (or luck) denies mobility. */
  failure?: string;
  modifiers?: Record<string, number | string>;
}

interface SceneSpec {
  title: string;
  context: string;
  media: MediaKey;
  audio: TrackKey;
  cinematic?: boolean;
  year: number;
  decisions: DecisionSpec[];
}

const AUTO = 'Continuar';

const SCENES: Record<string, SceneSpec> = {
  // ---------------------------------------------------------------- Act I · Bello, 2026
  start: {
    title: '5:40 a. m. Llueve sobre Bello.',
    context:
      'La alarma suena antes que el gallo del vecino. Andrés ya salió a su primer turno; dejó el tinto servido y una nota: «Hoy sí sale el bono». ' +
      'Sofía, de cuatro años, duerme abrazada a un peluche sin un ojo. En la nevera, la pluma de insulina de tu mamá marca dos dosis. Dos. ' +
      'La siguiente cuesta 98.000 pesos y la EPS dice que «está en trámite» desde hace tres semanas.\n\n' +
      'Te pones el impermeable, te cuelgas la caja térmica a la espalda y abres la app. Una notificación te saluda antes que nadie: ' +
      '«Tu tasa de aceptación bajó al 81 %. Los aliados con menos del 80 % reciben menos pedidos».\n\n' +
      'No tienes contrato, ni salud, ni pensión. La app te llama «aliada independiente». Tú le dices trabajo.',
    media: 'rainStreet', audio: 'rain', cinematic: true, year: 2026,
    decisions: [{ text: 'Encender la moto', next: 'order' }],
  },
  order: {
    title: '11:47 p. m. El pedido que nadie quiere',
    context:
      'Llevas 17 horas en la calle y te quedan 61.000 pesos libres. La app ofrece un pedido en lo alto de la loma, donde la semana pasada ' +
      'le robaron la moto a un compañero: +35.000 de «tarifa dinámica». Si lo rechazas, tu tasa cae por debajo del 80 %.',
    media: 'motoLight', audio: 'street', year: 2026,
    decisions: [
      { text: 'Aceptarlo. La insulina de tu mamá no puede esperar.', next: 'robbery' },
      { text: 'Rechazarlo. Sofía necesita que vuelvas a casa.', next: 'paused' },
    ],
  },
  robbery: {
    title: 'La loma',
    context:
      'La dirección termina en una cancha sin luz. Nadie pidió nada: el pedido era el anzuelo. Dos muchachos, uno de no más de quince años, ' +
      'te piden la moto «sin boleta». No te resistes; piensas en Sofía.\n\n' +
      'Caminas dos horas hasta la casa con el casco en la mano, porque es lo único que te dejaron. En el chat de soporte, un bot responde a las 2:14 a. m.: ' +
      '«Lamentamos lo ocurrido. Recuerda que, como aliada independiente, la plataforma no responde por tus herramientas de trabajo».\n\n' +
      'A la moto todavía le faltaban cuatro cuotas.',
    media: 'cityTraffic', audio: 'street', cinematic: true, year: 2026,
    decisions: [{ text: AUTO, next: 'emergency' }],
  },
  paused: {
    title: 'Cuenta pausada',
    context:
      'Llegas a medianoche y Sofía se despierta solo para decir «mami» y volverse a dormir. A las 6:00 a. m. la app no carga. ' +
      'Un mensaje en gris: «Tu cuenta fue pausada temporalmente por baja tasa de aceptación. Podrás conectarte de nuevo en 72 horas».\n\n' +
      'No hay a quién llamar, ni un formulario para explicar que rechazar un pedido fue elegir vivir. Tres días sin ingresos. ' +
      'Cuentas las monedas del frasco del café: 23.400 pesos.',
    media: 'phoneNight', audio: 'rain', cinematic: true, year: 2026,
    decisions: [{ text: AUTO, next: 'emergency' }],
  },
  emergency: {
    title: 'Urgencias del San Vicente',
    context:
      'Rosa colapsó a las 4:15 p. m. Cetoacidosis, dice el médico: le faltó la insulina. La EPS autoriza la cama, pero no los medicamentos: ' +
      '«Vuelva en 15 días». En la farmacia piden 340.000 pesos ya. En la sala de espera, un hombre amable te ofrece «una ayudita» sin papeles.',
    media: 'erWheelchairs', audio: 'tension', year: 2026,
    decisions: [
      { text: 'Aceptar el préstamo gota a gota: 20.000 diarios durante 30 días.', next: 'loan' },
      { text: 'Tragarte la vergüenza y pedirle ayuda a la jefe de enfermería.', next: 'nurseHelp' },
      { text: 'Vender tu celular, la herramienta con la que trabajas.', next: 'phoneSold' },
    ],
  },
  loan: {
    title: 'Veinte mil diarios',
    context:
      'Rosa sale del hospital con su insulina. Desde entonces, todos los días a las 6:00 p. m., una moto se detiene frente a tu puerta. ' +
      'No golpean: esperan. Veinte mil diarios por 400.000 prestados; al final habrás pagado 600.000.\n\n' +
      'Doblas turnos, dejas de almorzar, aprendes a dormir cuatro horas. Un jueves te faltan 8.000 pesos. ' +
      'El cobrador mira a Sofía jugar en la acera y dice, sin levantar la voz: «Qué niña tan linda. Cuídela».\n\nEsa noche no duermes.',
    media: 'debtWall', audio: 'street', cinematic: true, year: 2026,
    decisions: [{ text: AUTO, next: 'talk' }],
  },
  nurseHelp: {
    title: 'La jefe Marta',
    context:
      'La jefe de enfermería te reconoce: fuiste su estudiante hace cinco años, la del primer puesto, la que dejó la carrera en cuarto semestre ' +
      'cuando llegó Sofía. No te juzga. Consigue muestras de insulina, llama a la trabajadora social y te deja un papel en la mano: ' +
      'el hospital abre cupos para terminar enfermería en jornada nocturna.\n\n' +
      '«Usted tenía manos para esto, mija. Todavía las tiene».\n\n' +
      'Por primera vez en meses, alguien te mira como si tuvieras futuro.',
    media: 'insulin', audio: 'piano', cinematic: true, year: 2026,
    decisions: [{ text: AUTO, next: 'talk' }],
  },
  phoneSold: {
    title: 'Sin pantalla',
    context:
      'En la compraventa te dan 280.000 por el celular que te costó 900.000. Rosa recibe su insulina. ' +
      'Tú te quedas sin mapa, sin app, sin manera de trabajar.\n\n' +
      'Andrés reparte por los dos: quince horas, dieciséis, diecisiete. Llega con los ojos rojos y las manos temblando. ' +
      'Una noche, mientras cuenta monedas en la mesa de la cocina, te dice sin mirarte: «Así no vamos a salir nunca».',
    media: 'bogotaNight', audio: 'rain', cinematic: true, year: 2026,
    decisions: [{ text: AUTO, next: 'talk' }],
  },
  talk: {
    title: '3:00 a. m. La conversación',
    context:
      'Andrés te muestra un video: familias cruzando el Darién con niños al hombro. «Mi primo llegó a Houston en tres semanas. ' +
      'Allá un repartidor gana en un día lo que yo aquí en un mes». Habla del coyote, de la deuda, de mandar dólares. ' +
      'Tiene miedo, pero más miedo le da quedarse.',
    media: 'medellinDawn', audio: 'piano', year: 2026,
    decisions: [
      { text: 'Pedirle que se quede: aquí nos tenemos los tres.', next: 'together' },
      { text: 'Dejarlo ir: alguien tiene que intentarlo.', next: 'alone' },
      { text: 'Irse los tres por el Darién, con Sofía.', next: 'darien' },
    ],
  },

  // ---------------------------------------------------------------- Act II · seven years later
  together: {
    title: 'Medellín, 2033. Él se quedó.',
    context:
      'Fueron años duros, pero juntos: dos ingresos, dos turnos, una niña de once años que quiere ser veterinaria. ' +
      'Retomaste enfermería de noche mientras trabajabas en un call center. Hoy hay dos caminos sobre la mesa y ninguno es seguro. ' +
      'Elige sabiendo que el mundo también elige por ti.',
    media: 'womanAhead', audio: 'piano', year: 2033,
    decisions: [
      { text: 'Graduarte de enfermería y concursar por la plaza del hospital.', next: 'nurseWin', failure: 'nurseLose', modifiers: MODIFIERS.nurseTogether },
      { text: 'Quedarte en el call center y aceptar el ascenso a supervisora.', next: 'supervisorWin', failure: 'supervisorLose', modifiers: MODIFIERS.supervisorTogether },
      { text: 'Volver a la moto: es lo único seguro que conoces.', next: 'wheel' },
    ],
  },
  alone: {
    title: 'Medellín, 2033. Siete años de videollamadas.',
    context:
      'Andrés llegó a Houston. Las remesas llegan cuando hay trabajo y desaparecen cuando no. Sofía tiene once años y conoce a su papá ' +
      'por una pantalla. Sostienes la casa, a tu mamá y a tu hija. Terminaste enfermería a punta de madrugadas. ' +
      'Ahora debes decidir cómo sobrevivir sola.',
    media: 'planeSunset', audio: 'piano', year: 2033,
    decisions: [
      { text: 'Concursar por la plaza de enfermera del hospital.', next: 'againstOdds', failure: 'unseen', modifiers: MODIFIERS.nurseAlone },
      { text: 'Aceptar el trabajo de interna en una casa de El Poblado.', next: 'againstOdds', failure: 'otherHouse', modifiers: MODIFIERS.liveInAlone },
    ],
  },
  darien: {
    title: 'El Darién, 2027',
    context:
      'Diez días de selva. El barro llega a las rodillas y el río se lleva las mochilas de quienes resbalan. ' +
      'Andrés carga a Sofía en los hombros; tú cargas el agua y el miedo. Ves cosas que no le contarás a nadie: ' +
      'carpas abandonadas, zapatos pequeños sin dueño, una mujer que se sienta en una piedra y ya no se levanta.\n\n' +
      'Pagas a coyotes distintos en Panamá, en Costa Rica, en México: cada frontera cobra su peaje. ' +
      'Llegan a Houston en marzo, con 40 dólares y una cita con un abogado que cobra 3.000.\n\n' +
      'Están vivos. Están juntos. En un país que no los quiere.',
    media: 'riverVideo', audio: 'rainforest', cinematic: true, year: 2027,
    decisions: [{ text: AUTO, next: 'houston' }],
  },
  houston: {
    title: 'Houston, 2033',
    context:
      'Seis años con el asilo en trámite. Andrés trabaja en construcción. Sofía habla inglés mejor que ustedes y traduce en las citas médicas. ' +
      'Tú limpias oficinas de noche. Un community college abre un programa de técnica de enfermería para inmigrantes.',
    media: 'houstonNight', audio: 'street', year: 2033,
    decisions: [
      { text: 'Inscribirte en el programa de técnica de enfermería.', next: 'technicianWin', failure: 'technicianLose', modifiers: MODIFIERS.technicianHouston },
      { text: 'Seguir limpiando oficinas: no puedes arriesgar el sueldo.', next: 'againstOdds', failure: 'nightShift', modifiers: MODIFIERS.cleanerHouston },
    ],
  },

  // ---------------------------------------------------------------- outcomes (2033)
  nurseWin: {
    title: 'Bata blanca',
    context:
      'Ganaste la plaza. El día que te entregan el carné, Rosa llora más que tú. Atiendes urgencias en el mismo hospital donde, hace siete años, ' +
      'te ofrecieron un préstamo gota a gota. Cada noche llega alguien con la mirada que tú tenías: la de quien cuenta monedas frente a una farmacia.\n\n' +
      'El modelo estimaba tu salida con un 43 % de probabilidad. No fue solo mérito: fue un hogar con dos ingresos, una mano tendida ' +
      'y una moneda que cayó del lado correcto. ¿Cuántas Danielas igual de capaces cayeron del otro?',
    media: 'graduation', audio: 'hope', cinematic: true, year: 2033,
    decisions: [{ text: AUTO, next: 'end' }],
  },
  nurseLose: {
    title: 'La plaza fue para otra',
    context:
      'Pasaste el examen con buena nota, pero la plaza fue para alguien «con mejor perfil»: sin hijos, sin turnos partidos, con un tío en la junta. ' +
      'Sigues de auxiliar con un contrato de prestación de servicios que se renueva cada tres meses.\n\n' +
      'Hiciste todo bien. El modelo te daba un 43 %: más que a la mayoría, menos que a quien nació en otro barrio. ' +
      'Esa es la parte de la injusticia que no aparece en ninguna hoja de vida.',
    media: 'erEntrance', audio: 'piano', cinematic: true, year: 2033,
    decisions: [{ text: AUTO, next: 'end' }],
  },
  supervisorWin: {
    title: 'Supervisora',
    context:
      'Tienes contrato indefinido, cesantías y un escritorio con tu nombre. Supervisas a treinta muchachos que contestan quejas en inglés ' +
      'por un salario que no les alcanza para el arriendo. Les ves la misma prisa que tú tenías en la moto.\n\n' +
      'Saliste de la calle con un 37 % de probabilidad. A veces, en el bus de vuelta, te preguntas si la escalera la subiste tú ' +
      'o si fue el azar el que te empujó.',
    media: 'callCenter', audio: 'hope', cinematic: true, year: 2033,
    decisions: [{ text: AUTO, next: 'end' }],
  },
  supervisorLose: {
    title: 'Recorte de personal',
    context:
      'La empresa trasladó la operación a otro país, donde el minuto de llamada sale más barato. Doscientas personas a la calle un lunes, ' +
      'con una caja de cartón y un correo automático. El ascenso prometido nunca llegó.\n\n' +
      'El modelo te daba un 37 %. Andrés vuelve a sacar la moto del garaje. «Por lo menos estamos juntos», dice. Y es verdad. Pero no alcanza.',
    media: 'cityTraffic', audio: 'piano', cinematic: true, year: 2033,
    decisions: [{ text: AUTO, next: 'end' }],
  },
  wheel: {
    title: 'La rueda',
    context:
      'Volviste a la moto. La app ahora se llama distinto, pero las reglas son las mismas: tarifas que bajan, algoritmos que castigan, ' +
      'cero seguridad social. Andrés y tú se turnan la misma moto, uno de día y otro de noche. Sofía hace las tareas sola.\n\n' +
      'Nadie decidió que fuera así; simplemente nadie decidió lo contrario. En Colombia, más de la mitad de quienes trabajan ' +
      'lo hacen en la informalidad. Tú eres una de ellas.',
    media: 'timeMoney', audio: 'street', cinematic: true, year: 2033,
    decisions: [{ text: AUTO, next: 'end' }],
  },
  unseen: {
    title: 'Lo que el modelo no vio',
    context:
      'Concursaste por la plaza con el mejor puntaje de tu promoción. No la ganaste. Ni esa, ni la siguiente.\n\n' +
      'Para el modelo, una mujer sola con una hija y una madre enferma tenía un 8 % de probabilidad de salir adelante: tan poco ' +
      'que ni siquiera lanzó la moneda. No importó cuánto estudiaste. Lo que el algoritmo aprendió del mundo es que el mundo ' +
      'no espera a las mujeres que cuidan solas. Esta historia existe para preguntarte si estás de acuerdo.',
    media: 'rainWindow', audio: 'piano', cinematic: true, year: 2033,
    decisions: [{ text: AUTO, next: 'end' }],
  },
  otherHouse: {
    title: 'La casa ajena',
    context:
      'Cocinas, limpias y cuidas a dos niños que no son tuyos en una casa con vista a toda la ciudad. Ves a Sofía los domingos, de 10 a 6. ' +
      'Ella ya no te cuenta sus cosas; se las cuenta a la abuela. La señora dice que eres «como de la familia», pero comes sola en la cocina.\n\n' +
      'El modelo te daba un 1 %: ninguna moneda podía cambiar ese destino. El trabajo de cuidado sostiene al mundo, y el mundo apenas lo paga.',
    media: 'kitchen', audio: 'piano', cinematic: true, year: 2033,
    decisions: [{ text: AUTO, next: 'end' }],
  },
  againstOdds: {
    title: 'Contra la estadística',
    context:
      'Esto no debía pasar. El modelo casi no te daba oportunidad y aun así saliste adelante. Las historias como la tuya existen, ' +
      'y por eso las repetimos tanto: nos dejan creer que el sistema es justo.\n\n' +
      'Pero por cada historia como esta hay cientos de mujeres con tu mismo perfil, tu misma fuerza y tus mismas noches sin dormir, ' +
      'que nadie va a contar.',
    media: 'forestLight', audio: 'hope', cinematic: true, year: 2033,
    decisions: [{ text: AUTO, next: 'end' }],
  },
  technicianWin: {
    title: 'Técnica en Houston',
    context:
      'Terminaste el programa con honores, estudiando de día y limpiando oficinas de noche. Trabajas en un hospital de Houston y ganas ' +
      'en una semana lo que en Medellín ganabas en tres meses. Sofía va a ir a la universidad.\n\n' +
      'Pero tu mamá murió en Bello en 2030 y no pudiste ir al entierro: si salías del país, no podías volver. El modelo te daba un 26 %. ' +
      'Ganaste. Y aun así hay pérdidas que ninguna probabilidad mide.',
    media: 'houstonNight', audio: 'hope', cinematic: true, year: 2033,
    decisions: [{ text: AUTO, next: 'end' }],
  },
  technicianLose: {
    title: 'Sin papeles',
    context:
      'Te aceptaron en el programa, pero un cambio en las reglas de asilo dejó tu caso en el aire. Sin permiso de trabajo no puedes hacer ' +
      'las prácticas. Sigues limpiando oficinas, ahora con miedo a cada patrulla.\n\n' +
      'El modelo te daba un 26 % y la moneda cayó del otro lado. Cruzaste una selva con tu hija en brazos para descubrir ' +
      'que algunas fronteras no terminan nunca.',
    media: 'jungleDark', audio: 'piano', cinematic: true, year: 2033,
    decisions: [{ text: AUTO, next: 'end' }],
  },
  nightShift: {
    title: 'El turno de noche',
    context:
      'Limpias la misma oficina desde hace seis años. Conoces cada escritorio: las fotos familiares, los post-its, las tazas con frases ' +
      'motivadoras. Los dueños de esos escritorios nunca te han visto la cara.\n\n' +
      'Para el modelo, tu probabilidad era del 6 %: ni siquiera hubo moneda. Trabajas cuando la ciudad duerme y duermes cuando ' +
      'tu hija está despierta. Es el mismo turno de noche que empezaste en Bello, en otro idioma.',
    media: 'cleanOffice', audio: 'piano', cinematic: true, year: 2033,
    decisions: [{ text: AUTO, next: 'end' }],
  },

  // ---------------------------------------------------------------- shared closing
  end: {
    title: 'Turno de Noche',
    context:
      'Daniela no existe, pero millones de mujeres viven su historia. Cada decisión que tomaste fue razonable; ninguna bastó por sí sola. ' +
      'La IA no decidió tu destino: midió el peso del mundo que la entrenó. ¿Qué harías distinto? ¿Y qué nunca dependió de ti?',
    media: 'forestLight', audio: 'hope', year: 2033,
    decisions: [],
  },
};

/** Context budgets that fit the game's layout (frontend/src/scenes/Decision.js fits the text above the choices). */
const MAX_CONTEXT = { cinematic: 1000, other: 420 };

/** Seconds a scene holds before its choices (or the next scene) appear: enough time to read its text. */
function timerFor(scene: SceneSpec): number {
  const reading = scene.context.length + scene.title.length;
  if (scene.decisions.length === 0) return 8;
  return scene.cinematic ? Math.min(60, Math.max(10, Math.round(reading / 15) + 3)) : Math.min(18, Math.max(5, Math.round(reading / 25)));
}

/** Checks the narrative graph before anything is written: every target exists and every text fits. */
function validateGraph(): void {
  for (const [key, scene] of Object.entries(SCENES)) {
    const limit = scene.cinematic ? MAX_CONTEXT.cinematic : MAX_CONTEXT.other;
    if (scene.title.length > 120) throw new Error(`${key}: title longer than 120 characters`);
    if (scene.context.length > limit) throw new Error(`${key}: context is ${scene.context.length} characters (max ${limit})`);
    for (const decision of scene.decisions) {
      for (const target of [decision.next, decision.failure].filter(Boolean) as string[]) {
        if (!(target in SCENES)) throw new Error(`${key}: decision "${decision.text}" leads to unknown scene "${target}"`);
      }
      if (decision.text.length > 255) throw new Error(`${key}: decision text longer than 255 characters`);
    }
  }
}

type Created = { uid: string; documentId: string };

/**
 * Creates the story when it does not exist yet.
 *
 * @param strapi - the Strapi instance (called from bootstrap).
 */
export async function seedTurnoDeNoche(strapi: Core.Strapi): Promise<void> {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const docs = (uid: string) => strapi.documents(uid as any) as any;

  const existing = await docs('api::story.story').findFirst({ filters: { title: STORY_TITLE }, fields: ['title'] });
  if (existing) {
    strapi.log.info(`[seed] "${STORY_TITLE}" already exists; nothing to do`);
    return;
  }
  validateGraph();

  // Media library ids, by unique file name; missing files (other environments) become null.
  const media: Partial<Record<MediaKey, number>> = {};
  for (const [key, name] of Object.entries(MEDIA) as [MediaKey, string][]) {
    const file = await strapi.db.query('plugin::upload.file').findOne({ where: { name }, select: ['id'] });
    if (file) media[key] = file.id;
  }
  const missing = Object.keys(MEDIA).filter((key) => media[key as MediaKey] === undefined);
  if (missing.length > 0) {
    strapi.log.warn(`[seed] "${STORY_TITLE}": ${missing.length} media file(s) not found, scenes will use none: ${missing.join(', ')}`);
  }

  const created: Created[] = [];
  const create = async (uid: string, data: Record<string, unknown>) => {
    const document = await docs(uid).create({ data });
    created.push({ uid, documentId: document.documentId });
    return document.documentId as string;
  };

  try {
    const tracks: Partial<Record<TrackKey, string>> = {};
    for (const [key, track] of Object.entries(AUDIO_TRACKS) as [TrackKey, (typeof AUDIO_TRACKS)[TrackKey]][]) {
      const fileId = media[track.file];
      if (fileId === undefined) continue;
      const found = await docs('api::audio-track.audio-track').findFirst({ filters: { name: track.name }, fields: ['name'] });
      tracks[key] = found
        ? found.documentId
        : await create('api::audio-track.audio-track', { name: track.name, audio_file: fileId, loop: true, default_volume: track.volume });
    }

    const storyId = await create('api::story.story', {
      title: STORY_TITLE,
      short_description:
        'Daniela reparte comida por una app para sostener a su hija y a su madre diabética. Una noche de lluvia, una emergencia la obliga a elegir entre su salud, su dignidad y su futuro.',
      description:
        'Medellín, 2026. Una historia sobre el trabajo en plataformas, las barreras del sistema de salud, los préstamos gota a gota, ' +
        'el cuidado que recae sobre las mujeres y la migración por el Darién. Cada prueba de movilidad evalúa con la IA la vida que abre ' +
        'tu decisión, siete años después.',
      year: 2026,
      base_profile: BASE_PROFILE,
      cover_image: media.motoLight ?? null,
    });

    const sceneIds: Record<string, string> = {};
    for (const [key, scene] of Object.entries(SCENES)) {
      sceneIds[key] = await create('api::scene.scene', {
        title: scene.title,
        context: scene.context,
        background_media: media[scene.media] ?? null,
        audio_track: tracks[scene.audio] ?? null,
        timer_duration: timerFor(scene),
        is_cinematic: scene.cinematic ?? false,
        year: scene.year,
        story: storyId,
      });
    }

    // Decisions are created in display order (the BFF sorts them by id).
    for (const [key, scene] of Object.entries(SCENES)) {
      for (const decision of scene.decisions) {
        await create('api::decision.decision', {
          text: decision.text,
          requires_mobility_check: decision.failure !== undefined,
          hidden_modifiers: decision.modifiers ?? null,
          source_scene: sceneIds[key],
          next_scene: sceneIds[decision.next],
          failure_scene: decision.failure ? sceneIds[decision.failure] : null,
        });
      }
    }

    await docs('api::story.story').update({ documentId: storyId, data: { starting_scene: sceneIds.start } });
    strapi.log.info(
      `[seed] created "${STORY_TITLE}": ${Object.keys(SCENES).length} scenes, ` +
        `${created.filter((c) => c.uid === 'api::decision.decision').length} decisions, ${Object.keys(tracks).length} audio tracks`,
    );
  } catch (error) {
    // All or nothing: remove what was created, newest first, so the next start can try again cleanly.
    for (const { uid, documentId } of created.reverse()) {
      await docs(uid).delete({ documentId }).catch(() => undefined);
    }
    throw error;
  }
}
