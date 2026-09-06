/** Injection token for RandomSource. */
export const RANDOM_SOURCE = Symbol('RandomSource');

/**
 * Outbound port: a source of uniform random numbers for systemic luck.
 * Injected so tests can use a fixed sequence and production a secure generator.
 */
export interface RandomSource {
  /** A uniformly distributed number in [0, 1). */
  next(): number;
}
