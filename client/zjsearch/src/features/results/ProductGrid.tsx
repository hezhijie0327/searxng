// SPDX-License-Identifier: Apache-2.0 WITH Commons-Clause-1.0

import { Globe, Package, Tag, Truck } from "lucide-react";
import { ResultLink } from "@/features/results/cardParts.tsx";
import { TileEngines, TileThumb } from "@/features/results/Tile.tsx";
import { SWIPE_ROW } from "@/lib/styles.ts";
import type { GlobalData, ResultItem } from "@/lib/types.ts";

export function ProductGrid({ results, globals }: { results: ResultItem[]; globals: GlobalData }) {
  return (
    <div className="grid grid-cols-2 gap-x-4 gap-y-8 sm:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5">
      {results.map((result, index) => (
        <article className="group flex flex-col" key={`${result.url}-${index}`}>
          <ResultLink
            className="relative block aspect-square overflow-hidden rounded-xl bg-surface-2"
            globals={globals}
            result={result}
          >
            <TileThumb
              alt={result.title_text}
              placeholder={
                <span className="grid size-full place-items-center text-ink-3">
                  <Package className="size-8" />
                </span>
              }
              src={result.thumbnail}
            />
          </ResultLink>
          <h3 className="mt-2.5 line-clamp-2 text-base font-medium leading-snug">
            <ResultLink
              className="text-ink decoration-accent/50 underline-offset-2 hover:text-accent hover:underline"
              globals={globals}
              result={result}
            >
              <span dangerouslySetInnerHTML={{ __html: result.title_html }} dir="auto" />
            </ResultLink>
          </h3>
          {result.price ? (
            <p className="mt-1 inline-flex items-center gap-1 text-sm font-semibold text-ink">
              <Tag className="size-3.5 shrink-0 text-ink-3" />
              {result.price}
            </p>
          ) : null}
          <div className={`mt-0.5 text-xs text-ink-3 ${SWIPE_ROW} gap-x-2`}>
            {result.shipping ? (
              <span className="inline-flex items-center gap-1">
                <Truck className="size-3" />
                {result.shipping}
              </span>
            ) : null}
            {result.source_country ? (
              <span className="inline-flex items-center gap-1">
                <Globe className="size-3" />
                {result.source_country}
              </span>
            ) : null}
          </div>
          <div className="mt-auto pt-1.5">
            <TileEngines result={result} />
          </div>
        </article>
      ))}
    </div>
  );
}
