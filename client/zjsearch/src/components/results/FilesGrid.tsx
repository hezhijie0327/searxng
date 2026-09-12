// SPDX-License-Identifier: AGPL-3.0-or-later

/** Files-intent layout: square "file tiles" mirroring the media grids.
    Torrents and file downloads have no cover art, so the tile shows a
    type icon with the detected extension, the filesize takes the duration
    badge slot, and the magnet link is the card's inline primary action. */

import { formatDate } from "../../lib/format.ts";
import { useT } from "../../lib/i18n.ts";
import type { GlobalData, ResultItem } from "../../lib/types.ts";
import { ArrowDownIcon, ArrowUpIcon, DownloadIcon, FileIcon, FilmIcon, MagnetIcon, MusicIcon } from "../icons.tsx";
import { ResultLink, THEME_STATIC } from "./cards.tsx";

function detectExtension(title: string): string | null {
  const match = /\.([a-z0-9]{1,4})$/i.exec(title.trim());
  return match?.[1]?.toUpperCase() ?? null;
}

function TileIcon({ result }: { result: ResultItem }) {
  if (result.mtype === "audio") {
    return <MusicIcon className="size-10" />;
  }
  if (result.mtype === "video") {
    return <FilmIcon className="size-10" />;
  }
  return <FileIcon className="size-10" />;
}

export function FilesGrid({
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
  const cells = results.map((result, index) => {
    const extension = detectExtension(result.filename || result.title_text);
    const size = result.filesize || result.size || null;
    const date = result.published_date ? formatDate(result.published_date) : result.time || null;
    const hasHealth = result.seed !== undefined || result.leech !== undefined;
    const hotkeyIndex = indexOffset + index;
    const downloadHref =
      result.torrentfile ||
      (result.embedded && result.mtype !== "audio" && result.mtype !== "video" ? result.embedded : null);
    const hasActions = Boolean(result.magnetlink || downloadHref);
    return (
      <article
        className={`group rounded-2xl ${selected === hotkeyIndex ? "bg-surface ring-1 ring-accent-strong" : ""}`}
        data-hotkey-index={hotkeyIndex}
        key={`${result.url}-${index}`}
      >
        <ResultLink
          className="relative block aspect-square overflow-hidden rounded-xl border border-line bg-gradient-to-br from-surface-2 to-surface"
          globals={globals}
          result={result}
        >
          <span className="absolute inset-0 flex flex-col items-center justify-center gap-2 text-ink-3">
            <TileIcon result={result} />
            {extension ? (
              <span className="text-[11px] font-semibold uppercase tracking-widest">{extension}</span>
            ) : null}
          </span>
          {size ? (
            <span className="absolute bottom-2 right-2 rounded bg-black/80 px-1.5 py-0.5 text-[11px] font-medium text-white">
              {size}
            </span>
          ) : null}
          {result.favicon ? (
            <img
              alt=""
              className="absolute bottom-2 left-2 size-6 rounded-full bg-white ring-1 ring-white/25"
              decoding="async"
              loading="lazy"
              onError={(event) => {
                event.currentTarget.src = `${THEME_STATIC}/img/empty_favicon.svg`;
              }}
              src={result.favicon}
            />
          ) : null}
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
        <div className="mt-1.5 flex items-center justify-between gap-2 text-xs text-ink-3">
          {hasHealth ? (
            <span className="inline-flex items-center gap-2">
              {result.seed !== undefined ? (
                <span className="inline-flex items-center gap-0.5 font-medium text-ok">
                  <ArrowUpIcon className="size-3" />
                  {result.seed}
                </span>
              ) : null}
              {result.leech !== undefined ? (
                <span className="inline-flex items-center gap-0.5 font-medium text-danger">
                  <ArrowDownIcon className="size-3" />
                  {result.leech}
                </span>
              ) : null}
            </span>
          ) : (
            <span className="truncate" dir="auto">
              {result.engines[0]}
            </span>
          )}
          <span className="shrink-0">{date}</span>
        </div>
        {hasHealth || hasActions ? (
          <div className="mt-1.5 flex items-center justify-between gap-2">
            <span className="truncate text-xs text-ink-3">{hasHealth ? result.engines[0] : ""}</span>
            {hasActions ? (
              <span className="flex shrink-0 items-center gap-1.5">
                {result.magnetlink ? (
                  <a
                    aria-label={t("magnet_link")}
                    className="grid size-7 place-items-center rounded-full bg-accent-soft text-accent transition-colors hover:bg-accent-strong hover:text-accent-contrast"
                    href={result.magnetlink}
                    title={t("magnet_link")}
                  >
                    <MagnetIcon className="size-3.5" />
                  </a>
                ) : null}
                {downloadHref ? (
                  <a
                    aria-label={result.torrentfile ? t("torrent_file") : t("download")}
                    className="grid size-7 place-items-center rounded-full bg-surface-2 text-ink-2 transition-colors hover:text-ink"
                    download
                    href={downloadHref}
                    rel="noreferrer"
                    target="_blank"
                    title={result.torrentfile ? t("torrent_file") : t("download")}
                  >
                    <DownloadIcon className="size-3.5" />
                  </a>
                ) : null}
              </span>
            ) : null}
          </div>
        ) : null}
      </article>
    );
  });
  return <div className="grid grid-cols-2 gap-x-4 gap-y-8 sm:grid-cols-3 xl:grid-cols-4">{cells}</div>;
}
