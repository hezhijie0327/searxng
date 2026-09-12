// SPDX-License-Identifier: AGPL-3.0-or-later

import { useState } from "react";
import { THEME_STATIC } from "../../lib/constants.ts";
import { formatDate, formatLength } from "../../lib/format.ts";
import { useT } from "../../lib/i18n.ts";
import type { GlobalData, ResultItem } from "../../lib/types.ts";
import { CalendarIcon, CloseIcon, ImageIcon, PackageIcon, PlayIcon } from "../icons.tsx";
import { ResultLink } from "./cardParts.tsx";
import { TileBadge, TileFavicon } from "./Tile.tsx";

// ------------------------------------------------------------- grid layouts

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
            {result.thumbnail ? (
              <img
                alt={result.title_text}
                className="size-full object-cover transition-transform duration-300 group-hover:scale-[1.03]"
                decoding="async"
                loading="lazy"
                onError={(event) => {
                  event.currentTarget.src = `${THEME_STATIC}/img/img_load_error.svg`;
                }}
                src={result.thumbnail}
              />
            ) : (
              <span className="grid size-full place-items-center text-ink-3">
                <PackageIcon className="size-8" />
              </span>
            )}
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
          {result.price ? <p className="mt-1 text-sm font-semibold text-ink">{result.price}</p> : null}
          <div className="mt-0.5 flex flex-wrap items-center gap-x-2 text-xs text-ink-3">
            {result.shipping ? <span>{result.shipping}</span> : null}
            {result.source_country ? <span>{result.source_country}</span> : null}
            <span className="truncate">{result.engines[0]}</span>
          </div>
        </article>
      ))}
    </div>
  );
}

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
        className={`group -m-2 rounded-2xl p-2 ${selected === hotkeyIndex ? "bg-surface ring-1 ring-accent-strong" : ""}`}
        data-hotkey-index={hotkeyIndex}
        key={`${result.url}-${index}`}
      >
        <div className="relative">
          <ResultLink
            className="relative block aspect-video overflow-hidden rounded-xl bg-surface-2"
            globals={globals}
            result={result}
          >
            {result.thumbnail ? (
              <img
                alt={result.title_text}
                className="size-full object-cover transition-transform duration-300 group-hover:scale-[1.03]"
                decoding="async"
                loading="lazy"
                onError={(event) => {
                  event.currentTarget.src = `${THEME_STATIC}/img/img_load_error.svg`;
                }}
                src={result.thumbnail}
              />
            ) : (
              <span className="grid size-full place-items-center text-ink-3">
                <PlayIcon className="size-8" />
              </span>
            )}
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
              className="absolute end-2 top-2 z-20 grid size-7 place-items-center rounded-full bg-black/70 text-white transition-colors hover:bg-accent-strong hover:text-ink"
              onClick={() => {
                setPlaying(null);
              }}
              title={t("hide_video")}
              type="button"
            >
              <CloseIcon className="size-3.5" />
            </button>
          ) : result.iframe_src ? (
            <button
              aria-label={t("play")}
              className="absolute left-1/2 top-1/2 z-10 grid size-12 -translate-x-1/2 -translate-y-1/2 place-items-center rounded-full bg-black/60 text-white opacity-85 shadow-pop transition-all hover:scale-105 hover:bg-accent-strong hover:text-ink group-hover:opacity-100"
              onClick={() => {
                setPlaying(index);
              }}
              title={t("play")}
              type="button"
            >
              <PlayIcon className="size-5 translate-x-px" />
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
          <span className="truncate" dir="auto">
            {result.author || result.engines[0]}
          </span>
          <span className="flex shrink-0 items-center gap-2">
            {result.views ? <span>{result.views}</span> : null}
            <span className="flex items-center gap-1">
              {result.published_date ? (
                <>
                  <CalendarIcon className="size-3" />
                  {formatDate(result.published_date)}
                </>
              ) : null}
            </span>
          </span>
        </div>
      </article>
    );
  });
  return <div className="grid grid-cols-1 gap-x-4 gap-y-8 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">{cells}</div>;
}

/** Poster wall for the movies category (`!movies`, tmdb / imdb): portrait
    posters in a denser grid, mirroring the video grid's tile language. */
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
        className={`group -m-2 rounded-2xl p-2 ${selected === hotkeyIndex ? "bg-surface ring-1 ring-accent-strong" : ""}`}
        data-hotkey-index={hotkeyIndex}
        key={`${result.url}-${index}`}
      >
        <ResultLink
          className="relative block aspect-[2/3] overflow-hidden rounded-xl bg-surface-2"
          globals={globals}
          result={result}
        >
          {result.thumbnail ? (
            <img
              alt={result.title_text}
              className="size-full object-cover transition-transform duration-300 group-hover:scale-[1.03]"
              decoding="async"
              loading="lazy"
              onError={(event) => {
                event.currentTarget.src = `${THEME_STATIC}/img/img_load_error.svg`;
              }}
              src={result.thumbnail}
            />
          ) : (
            <span className="grid size-full place-items-center text-ink-3">
              <PlayIcon className="size-8" />
            </span>
          )}
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
      </article>
    );
  });
  return (
    <div className="grid grid-cols-2 gap-x-4 gap-y-8 sm:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 2xl:grid-cols-6">
      {cells}
    </div>
  );
}

/** App-store tiles for the apps category (google play, fdroid, apk mirror):
    square store icon beside the app name, short description below. */
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
        className={`group -m-2 rounded-2xl p-2 ${selected === hotkeyIndex ? "bg-surface ring-1 ring-accent-strong" : ""}`}
        data-hotkey-index={hotkeyIndex}
        key={`${result.url}-${index}`}
      >
        <div className="flex items-center gap-3">
          <ResultLink className="relative block shrink-0" globals={globals} result={result}>
            {result.thumbnail ? (
              <img
                alt={result.title_text}
                className="size-14 rounded-xl border border-line bg-surface-2 object-cover"
                decoding="async"
                loading="lazy"
                onError={(event) => {
                  event.currentTarget.src = `${THEME_STATIC}/img/img_load_error.svg`;
                }}
                src={result.thumbnail}
              />
            ) : (
              <span className="grid size-14 place-items-center rounded-xl border border-line bg-surface-2 text-ink-3">
                <ImageIcon className="size-6" />
              </span>
            )}
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
        <p className="mt-1.5 truncate text-xs text-ink-3">{result.engines[0]}</p>
      </article>
    );
  });
  return <div className="grid grid-cols-1 gap-x-4 gap-y-6 sm:grid-cols-2 xl:grid-cols-3">{cells}</div>;
}
