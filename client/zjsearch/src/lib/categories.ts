// SPDX-License-Identifier: Apache-2.0 WITH Commons-Clause-1.0

/**
 * Shared category vocabulary.  Server categories are open-ended strings
 * (engines declare their own), so there is no closed union type — these
 * constants only pin down the names the client special-cases.
 */

/** Pseudo category of engine-bang searches (`!imdb bat`): the results carry
    their real category and the page inherits its presentation from them. */
export const NO_CATEGORY = "none";

/** In mixed searches the packages results are presented inside the it block. */
export const PACKAGES_BLOCK_CATEGORY = "it";
