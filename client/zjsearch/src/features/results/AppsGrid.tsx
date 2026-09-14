// SPDX-License-Identifier: Apache-2.0 WITH Commons-Clause-1.0

/** App-store tiles for the apps category (google play, fdroid, apk mirror):
    square store icon beside the app name, short description below. */

import { Image } from "lucide-react";
import { ResultLink } from "@/features/results/cardParts.tsx";
import { TileEngines, TileThumb } from "@/features/results/Tile.tsx";
import type { GlobalData, ResultItem } from "@/lib/types.ts";

export function AppsGrid({
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
        <div className="flex items-center gap-3">
          <ResultLink className="relative block shrink-0" globals={globals} result={result}>
            <TileThumb
              alt={result.title_text}
              imgClassName="size-14 rounded-xl border border-line object-cover"
              placeholder={
                <span className="grid size-14 place-items-center rounded-xl border border-line bg-surface-2 text-ink-3">
                  <Image className="size-6" />
                </span>
              }
              src={result.thumbnail}
            />
          </ResultLink>
          <h3 className="line-clamp-2 min-h-[2.5rem] text-base font-medium leading-snug">
            <ResultLink
              className="text-ink decoration-accent/50 underline-offset-2 hover:text-accent hover:underline"
              globals={globals}
              result={result}
            >
              <span dangerouslySetInnerHTML={{ __html: result.title_html }} dir="auto" />
            </ResultLink>
          </h3>
        </div>
        {result.content_html ? (
          <p
            className="mt-1.5 line-clamp-2 text-xs leading-relaxed text-ink-2"
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
    <div className="grid grid-cols-1 gap-x-4 gap-y-6 @sm:grid-cols-2 @[46rem]:grid-cols-3 @5xl:grid-cols-4">
      {cells}
    </div>
  );
}
