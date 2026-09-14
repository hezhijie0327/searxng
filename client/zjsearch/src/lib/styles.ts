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

/** Multi-part meta row that swipes horizontally on overflow (same hidden
    scrollbar, children never shrink).  Callers add their own gap and text
    size (e.g. gap-x-3 text-xs text-ink-3). */
export const META_ROW = `flex items-center overflow-x-auto whitespace-nowrap ${SCROLLBAR_NONE} [&>*]:shrink-0`;

/** Circular ghost icon button, 36px with 18px icons — header actions,
    drawer/help closes, and every other chrome-level round button. */
export const ICON_BTN =
  "grid size-9 place-items-center rounded-full text-ink-2 transition-colors hover:bg-surface-2 hover:text-ink";

/** Tiny meta chip (engine pills, mono tokens, tag pills): callers add their
    own text colour / font / hover on top of the shape. */
export const CHIP = "inline-flex items-center gap-1 rounded-full bg-surface-2 px-2 py-0.5";
