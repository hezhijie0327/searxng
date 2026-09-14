// SPDX-License-Identifier: Apache-2.0 WITH Commons-Clause-1.0

/**
 * Shared category vocabulary.  Server categories are open-ended strings
 * (engines declare their own), so there is no closed union type — these
 * constants only pin down the names the client special-cases.
 */

/** Pseudo category of engine-bang searches (`!imdb bat`): the results carry
    their real category and the page inherits its presentation from them. */
export const NO_CATEGORY = "none";

/** Categories that stay visible in the tab row; everything else (and any
    future category) folds into the "more" menu. */
export const VISIBLE_CATEGORY_TABS: readonly string[] = ["general", "images", "videos", "news", "map", "music"];

/** In mixed searches the packages results are presented inside the it block. */
export const PACKAGES_BLOCK_CATEGORY = "it";
