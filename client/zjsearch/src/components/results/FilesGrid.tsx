// SPDX-License-Identifier: AGPL-3.0-or-later

/** Files-intent layout: landscape file tiles mirroring the media grids.
    Torrents and file downloads have no cover art, so the tile shows a
    type icon with the detected extension; the filesize takes the badge
    slot and the magnet/torrent action sits in the tile center like the
    media play buttons. */

import { formatDate } from "../../lib/format.ts";
import { useT } from "../../lib/i18n.ts";
import type { GlobalData, ResultItem } from "../../lib/types.ts";
import {
  ArrowDownIcon,
  ArrowUpIcon,
  CalendarIcon,
  DownloadIcon,
  FileIcon,
  FilmIcon,
  MagnetIcon,
  MusicIcon,
} from "../icons.tsx";
import { ResultLink, THEME_STATIC } from "./cards.tsx";

function detectExtension(title: string): string | null {
  const match = /\.([a-z0-9]{1,4})$/i.exec(title.trim());
  return match?.[1]?.toUpperCase() ?? null;
}

function TileIcon({ result, large = false }: { result: ResultItem; large?: boolean }) {
  const size = large ? "size-10" : "size-5";
  if (result.mtype === "audio") {
    return <MusicIcon className={size} />;
  }
  if (result.mtype === "video") {
    return <FilmIcon className={size} />;
  }
  return <FileIcon className={size} />;
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
  return (
    <div className="grid grid-cols-2 gap-x-4 gap-y-8 sm:grid-cols-3 xl:grid-cols-4">
      {results.map((result, index) => {
        const extension = detectExtension(result.filename || result.title_text);
        const size = result.filesize || result.size || null;
        const date = result.published_date ? formatDate(result.published_date) : result.time || null;
        const health = result.seed !== undefined || result.leech !== undefined;
        const downloadHref =
          result.torrentfile ||
          (result.embedded && result.mtype !== "audio" && result.mtype !== "video" ? result.embedded : null);
        // the tile's primary action: magnet when present, otherwise the
        // torrent file / direct download
        const primaryHref = result.magnetlink || downloadHref || null;
        const primaryLabel = result.magnetlink
          ? t("magnet_link")
          : result.torrentfile
            ? t("torrent_file")
            : t("download");
        const hotkeyIndex = indexOffset + index;
        return (
          <article
            className={`group rounded-2xl ${selected === hotkeyIndex ? "bg-surface ring-1 ring-accent-strong" : ""}`}
            data-hotkey-index={hotkeyIndex}
            key={`${result.url}-${index}`}
          >
            <div className="relative aspect-video overflow-hidden rounded-xl border border-line bg-gradient-to-br from-surface-2 to-surface">
              <span className="absolute left-3 top-3 flex items-center gap-1.5 text-ink-3">
                <TileIcon result={result} />
                {extension ? (
                  <span className="text-[10px] font-semibold uppercase tracking-widest">{extension}</span>
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
              {primaryHref ? (
                <a
                  aria-label={primaryLabel}
                  className="absolute left-1/2 top-1/2 z-10 grid size-12 -translate-x-1/2 -translate-y-1/2 place-items-center rounded-full bg-black/70 text-white shadow-pop transition-all hover:scale-105 hover:bg-accent-strong hover:text-ink"
                  href={primaryHref}
                  {...(result.magnetlink ? {} : { download: true })}
                  title={primaryLabel}
                >
                  {result.magnetlink ? <MagnetIcon className="size-5" /> : <DownloadIcon className="size-5" />}
                </a>
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
            <div className="mt-1.5 flex min-h-5 items-center justify-between gap-2 text-xs text-ink-3">
              {health ? (
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
              <span className="flex shrink-0 items-center gap-1">
                <CalendarIcon className="size-3" />
                {date}
              </span>
            </div>
            <div className="mt-1.5 flex min-h-5 items-center justify-between gap-2 text-xs text-ink-3">
              <span className="truncate">{health ? result.engines[0] : ""}</span>
              <span className="shrink-0" />
            </div>
          </article>
        );
      })}
    </div>
  );
}
