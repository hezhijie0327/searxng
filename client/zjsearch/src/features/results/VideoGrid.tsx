// SPDX-License-Identifier: Apache-2.0 WITH Commons-Clause-1.0

import { Calendar, Eye, Play, User, X } from "lucide-react";
import { useState } from "react";
import { ResultLink } from "@/features/results/cardParts.tsx";
import { TileBadge, TileEngines, TileFavicon, TileThumb } from "@/features/results/Tile.tsx";
import { formatDate, formatLength } from "@/lib/format.ts";
import { useT } from "@/lib/i18n.ts";
import type { GlobalData, ResultItem } from "@/lib/types.ts";

export function VideoGrid({
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
  const t = useT();
  const [playing, setPlaying] = useState<number | null>(null);
  const cells = results.map((result, index) => {
    const length = formatLength(result.length_display, result.length_seconds);
    const isPlaying = playing === index;
    const hotkeyIndex = indexOffset + index;
    return (
      <article
        className={`group -m-2 flex flex-col rounded-2xl p-2 ${selected === hotkeyIndex ? "bg-surface ring-1 ring-accent-strong" : ""}`}
        data-hotkey-index={hotkeyIndex}
        key={`${result.url}-${index}`}
      >
        <div className="relative">
          <ResultLink
            className="relative block aspect-video overflow-hidden rounded-xl bg-surface-2"
            globals={globals}
            result={result}
          >
            <TileThumb alt={result.title_text} placeholder={null} src={result.thumbnail} />
            {length ? <TileBadge>{length}</TileBadge> : null}
            {result.favicon ? <TileFavicon src={result.favicon} /> : null}
          </ResultLink>
          {result.iframe_src && isPlaying ? (
            <div className="absolute inset-0 z-10 animate-fade-in overflow-hidden rounded-xl border border-line bg-black">
              <iframe
                allowFullScreen
                className="size-full"
                referrerPolicy="origin"
                src={result.iframe_src ?? ""}
                title={result.title_text}
              />
            </div>
          ) : null}
          {result.iframe_src && isPlaying ? (
            <button
              aria-label={t("hide_video")}
              className="absolute end-2 top-2 z-20 grid size-7 place-items-center rounded-full bg-black/70 text-white transition-colors hover:bg-accent-strong hover:text-accent-contrast"
              onClick={() => {
                setPlaying(null);
              }}
              title={t("hide_video")}
              type="button"
            >
              <X className="size-3.5" />
            </button>
          ) : result.iframe_src ? (
            <button
              aria-label={t("play")}
              className="absolute left-1/2 top-1/2 z-10 grid size-12 -translate-x-1/2 -translate-y-1/2 place-items-center rounded-full bg-black/60 text-white opacity-85 shadow-pop transition-all hover:scale-105 hover:bg-accent-strong hover:text-accent-contrast group-hover:opacity-100"
              onClick={() => {
                setPlaying(index);
              }}
              title={t("play")}
              type="button"
            >
              <Play className="size-5 translate-x-px" />
            </button>
          ) : null}
        </div>
        <h3 className="mt-2.5 line-clamp-2 min-h-[2.75rem] text-base font-medium leading-snug">
          <ResultLink
            className="text-ink decoration-accent/50 underline-offset-2 hover:text-accent hover:underline"
            globals={globals}
            result={result}
          >
            <span dangerouslySetInnerHTML={{ __html: result.title_html }} dir="auto" />
          </ResultLink>
        </h3>
        <div className="mt-1.5 flex items-center justify-between gap-3 text-xs text-ink-3">
          {result.author ? (
            <span className="inline-flex min-w-0 items-center gap-1 truncate" dir="auto">
              <User className="size-3 shrink-0" />
              {result.author}
            </span>
          ) : (
            <span />
          )}
          <span className="flex shrink-0 items-center gap-2">
            {result.views ? (
              <span className="inline-flex items-center gap-1">
                <Eye className="size-3 shrink-0" />
                {result.views}
              </span>
            ) : null}
            <span className="flex items-center gap-1">
              {result.published_date ? (
                <>
                  <Calendar className="size-3" />
                  {formatDate(result.published_date)}
                </>
              ) : null}
            </span>
          </span>
        </div>
        <div className="mt-auto pt-1.5">
          <TileEngines result={result} />
        </div>
      </article>
    );
  });
  // density keys off the column width (container queries from the results
  // wrapper): widescreen grows to 4 columns, centered mode drops to 2-3
  return (
    <div className="grid grid-cols-1 gap-x-4 gap-y-8 @sm:grid-cols-2 @[46rem]:grid-cols-3 @[54rem]:grid-cols-4">
      {cells}
    </div>
  );
}
