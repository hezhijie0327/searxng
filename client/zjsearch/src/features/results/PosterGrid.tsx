// SPDX-License-Identifier: Apache-2.0 WITH Commons-Clause-1.0

/** Poster wall for the movies category (`!movies`, tmdb / imdb): portrait
    posters in a denser grid, mirroring the video grid's tile language. */

import { Clapperboard } from "lucide-react";
import { ResultLink } from "@/features/results/cardParts.tsx";
import { TileEngines, TileFavicon, TileThumb } from "@/features/results/Tile.tsx";
import type { GlobalData, ResultItem } from "@/lib/types.ts";

export function PosterGrid({
  results,
  globals,
  selected,
  indexOffset = 0,
}: {
  results: ResultItem[];
  globals: GlobalData;
  selected?: number;
  /** hotkey indices are page-global: offset by the grid's first result index */
  indexOffset?: number;
}) {
  const cells = results.map((result, index) => {
    const hotkeyIndex = indexOffset + index;
    return (
      <article
        className={`group -m-2 flex flex-col rounded-2xl p-2 ${selected === hotkeyIndex ? "bg-surface ring-1 ring-accent-strong" : ""}`}
        data-hotkey-index={hotkeyIndex}
        key={`${result.url}-${index}`}
      >
        <ResultLink
          className="relative block aspect-[2/3] overflow-hidden rounded-xl bg-surface-2"
          globals={globals}
          result={result}
        >
          <TileThumb
            alt={result.title_text}
            placeholder={
              <span className="grid size-full place-items-center text-ink-3">
                <Clapperboard className="size-8" />
              </span>
            }
            src={result.thumbnail}
          />
          {result.favicon ? <TileFavicon src={result.favicon} /> : null}
        </ResultLink>
        <h3 className="mt-2.5 line-clamp-2 min-h-[2.75rem] text-base font-medium leading-snug">
          <ResultLink
            className="text-ink decoration-accent/50 underline-offset-2 hover:text-accent hover:underline"
            globals={globals}
            result={result}
          >
            <span dangerouslySetInnerHTML={{ __html: result.title_html }} dir="auto" />
          </ResultLink>
        </h3>
        {result.content_html ? (
          <p
            className="mt-1 line-clamp-2 text-xs leading-relaxed text-ink-2"
            dangerouslySetInnerHTML={{ __html: result.content_html }}
            dir="auto"
          />
        ) : null}
        <div className="mt-auto pt-1.5">
          <TileEngines result={result} />
        </div>
      </article>
    );
  });
  return (
    <div className="grid grid-cols-2 gap-x-4 gap-y-8 sm:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 2xl:grid-cols-6">
      {cells}
    </div>
  );
}
