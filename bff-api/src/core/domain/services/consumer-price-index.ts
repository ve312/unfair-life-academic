import { InvalidStoryContentError } from '../errors/domain.error.js';

/**
 * Converts money between story years and the 1994 dollars the mobility model
 * was trained on, using the US CPI-U annual average (BLS series CUUR0000SA0,
 * 1982-84 = 100). Years after the last published value are projected at a
 * documented PROJECTION_RATE per year. The same table is reproduced in the
 * training notebook (notebooks/01_math_and_eda.ipynb) for the academic report.
 *
 *   CPI(t) = CPI_BLS(t)                                  t <= LAST_PUBLISHED_YEAR
 *   CPI(t) = CPI_BLS(LAST) * (1 + rate)^(t - LAST)       t >  LAST_PUBLISHED_YEAR
 *   x_1994 = x_t * CPI(1994) / CPI(t)
 */

/** Year of the census the model was trained on. */
export const MODEL_BASE_YEAR = 1994;
/** The model predicts "income above this amount" in MODEL_BASE_YEAR dollars. */
export const MODEL_INCOME_THRESHOLD_1994 = 50_000;
/** Annual inflation assumed for years without published CPI data. */
export const PROJECTION_RATE = 0.025;
/** Story years the economy supports (matches the CMS schema limits). */
export const STORY_YEAR_RANGE = { min: 1990, max: 2100 } as const;

const CPI_U_ANNUAL_AVERAGE: Readonly<Record<number, number>> = Object.freeze({
  1990: 130.7, 1991: 136.2, 1992: 140.3, 1993: 144.5, 1994: 148.2, 1995: 152.4, 1996: 156.9, 1997: 160.5,
  1998: 163.0, 1999: 166.6, 2000: 172.2, 2001: 177.1, 2002: 179.9, 2003: 184.0, 2004: 188.9, 2005: 195.3,
  2006: 201.6, 2007: 207.342, 2008: 215.303, 2009: 214.537, 2010: 218.056, 2011: 224.939, 2012: 229.594,
  2013: 232.957, 2014: 236.736, 2015: 237.017, 2016: 240.007, 2017: 245.12, 2018: 251.107, 2019: 255.657,
  2020: 258.811, 2021: 270.97, 2022: 292.655, 2023: 304.702, 2024: 313.689,
});

export const LAST_PUBLISHED_YEAR = Math.max(...Object.keys(CPI_U_ANNUAL_AVERAGE).map(Number));

export function isSupportedStoryYear(year: unknown): year is number {
  return Number.isInteger(year) && (year as number) >= STORY_YEAR_RANGE.min && (year as number) <= STORY_YEAR_RANGE.max;
}

function requireYear(year: number): void {
  if (!isSupportedStoryYear(year)) {
    throw new InvalidStoryContentError(
      `Story year must be an integer between ${STORY_YEAR_RANGE.min} and ${STORY_YEAR_RANGE.max}, got ${String(year)}`,
    );
  }
}

/** CPI-U for a year: published value, or projected from the last published year. */
export function consumerPriceIndex(year: number): number {
  requireYear(year);
  const published = CPI_U_ANNUAL_AVERAGE[year];
  if (published !== undefined) {
    return published;
  }
  return CPI_U_ANNUAL_AVERAGE[LAST_PUBLISHED_YEAR] * (1 + PROJECTION_RATE) ** (year - LAST_PUBLISHED_YEAR);
}

/** Whether the CPI value for a year is published (true) or projected (false). */
export function isPublishedYear(year: number): boolean {
  return CPI_U_ANNUAL_AVERAGE[year] !== undefined;
}

/** Converts an amount in `year` dollars to 1994 dollars. */
export function toModelBaseYearDollars(amount: number, year: number): number {
  return (amount * consumerPriceIndex(MODEL_BASE_YEAR)) / consumerPriceIndex(year);
}

/** The model's "$50K in 1994" expressed in `year` dollars, rounded to whole dollars. */
export function mobilityIncomeThreshold(year: number): number {
  return Math.round((MODEL_INCOME_THRESHOLD_1994 * consumerPriceIndex(year)) / consumerPriceIndex(MODEL_BASE_YEAR));
}
