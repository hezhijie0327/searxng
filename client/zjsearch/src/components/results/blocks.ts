// SPDX-License-Identifier: AGPL-3.0-or-later

import type { ResultItem } from "../../lib/types.ts";

/** Packages fold into the it block; every other category stands alone. */
function blockKeyOf(result: ResultItem): string {
  const category = result.category || "general";
  return category === "packages" ? "it" : category;
}

export function collectBlocks(results: ResultItem[]): Map<string, Array<{ result: ResultItem; index: number }>> {
  const blocks = new Map<string, Array<{ result: ResultItem; index: number }>>();
  results.forEach((result, index) => {
    const key = blockKeyOf(result);
    const items = blocks.get(key);
    if (items) {
      items.push({ result, index });
    } else {
      blocks.set(key, [{ result, index }]);
    }
  });
  return blocks;
}
