import { randomInt } from 'node:crypto';
import { Injectable } from '@nestjs/common';
import type { RandomSource } from '../../../core/ports/out/random-source.port.js';

/** randomInt's largest allowed range (max - min <= 2^48 - 1), giving ~48 bits of uniform precision. */
const RESOLUTION = 2 ** 48 - 1;

/**
 * RandomSource backed by the operating system's cryptographic generator, so
 * systemic-luck rolls cannot be predicted from earlier rolls.
 */
@Injectable()
export class CryptoRandomSource implements RandomSource {
  next(): number {
    // randomInt's upper bound is exclusive, so the result is in [0, 1).
    return randomInt(0, RESOLUTION) / RESOLUTION;
  }
}
