// SPDX-License-Identifier: Apache-2.0 WITH Commons-Clause-1.0

/**
 * Shared utility-class fragments for the result area's recurring visual
 * patterns.  Keeping the long Tailwind strings here means a tweak to the
 * swipe/scroll language updates every row at once.
 */

/** Hidden-scrollbar tail for horizontally swipeable rows (mobile-style). */
export const SCROLLBAR_NONE = "[scrollbar-width:none] [&::-webkit-scrollbar]:hidden";

/** Single-line swipe row: children never wrap or shrink — overflow swipes
    horizontally instead, like the mobile category tabs.  Callers add their
    own gap (e.g. gap-1 / gap-x-2). */
export const SWIPE_ROW = `flex flex-nowrap items-center overflow-x-auto ${SCROLLBAR_NONE} [&>*]:shrink-0`;
