// SPDX-License-Identifier: AGPL-3.0-or-later

/** Package results (crates.io, docker hub, pypi, ...): the file-tile
    language of FilesGrid applied to software packages - icon tile with the
    version badge, author/updated meta row, homepage/source actions. */

import { formatDate } from "../../lib/format.ts";
import { useT } from "../../lib/i18n.ts";
import type { GlobalData, ResultItem } from "../../lib/types.ts";
import { CodeIcon, ExternalLinkIcon, PackageIcon } from "../icons.tsx";
import { ResultLink, THEME_STATIC } from "./cards.tsx";
import { Strip } from "./Strip.tsx";

function PackageCell({
  result,
  globals,
  selected,
  hotkeyIndex,
}: {
  result: ResultItem;
  globals: GlobalData;
  selected?: number;
  hotkeyIndex: number;
}) {
  const t = useT();
  return (
    <article
      className={`group rounded-2xl ${selected === hotkeyIndex ? "bg-surface ring-1 ring-accent-strong" : ""}`}
      data-hotkey-index={hotkeyIndex}
    >
      <ResultLink
        className="relative block aspect-square overflow-hidden rounded-xl border border-line bg-gradient-to-br from-surface-2 to-surface"
        globals={globals}
        result={result}
      >
        <span className="absolute inset-0 flex flex-col items-center justify-center gap-2 text-ink-3">
          <PackageIcon className="size-10" />
          {result.version ? (
            <span className="font-mono text-[11px] font-semibold tracking-wide">{result.version}</span>
          ) : null}
        </span>
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
        <span className="truncate" dir="auto">
          {result.maintainer || result.author || result.engines[0]}
        </span>
        <span className="shrink-0">{result.published_date ? formatDate(result.published_date) : null}</span>
      </div>
      {result.homepage || result.source_code_url ? (
        <div className="mt-1.5 flex items-center gap-1.5">
          {result.homepage ? (
            <a
              aria-label={t("homepage")}
              className="grid size-7 place-items-center rounded-full bg-accent-soft text-accent transition-colors hover:bg-accent-strong hover:text-accent-contrast"
              href={result.homepage}
              rel="noreferrer"
              target="_blank"
              title={t("homepage")}
            >
              <ExternalLinkIcon className="size-3.5" />
            </a>
          ) : null}
          {result.source_code_url ? (
            <a
              aria-label={t("repository")}
              className="grid size-7 place-items-center rounded-full bg-surface-2 text-ink-2 transition-colors hover:text-ink"
              href={result.source_code_url}
              rel="noreferrer"
              target="_blank"
              title={t("repository")}
            >
              <CodeIcon className="size-3.5" />
            </a>
          ) : null}
          <span className="truncate text-xs text-ink-3">{result.engines[0]}</span>
        </div>
      ) : (
        <div className="mt-1.5 flex items-center">
          <span className="truncate text-xs text-ink-3">{result.engines[0]}</span>
        </div>
      )}
    </article>
  );
}

export function PackageGrid({
  results,
  globals,
  selected,
  indexOffset = 0,
  variant = "grid",
}: {
  results: ResultItem[];
  globals: GlobalData;
  selected?: number;
  /** hotkey indices are page-global: offset by the grid's first result index */
  indexOffset?: number;
  /** "strip" renders the same cells in a fixed-row horizontal carousel */
  variant?: "grid" | "strip";
}) {
  const cells = results.map((result, index) => (
    <PackageCell
      globals={globals}
      hotkeyIndex={indexOffset + index}
      key={`${result.url}-${index}`}
      result={result}
      selected={selected}
    />
  ));
  if (variant === "strip") {
    return <Strip rows={1}>{cells}</Strip>;
  }
  return <div className="grid grid-cols-2 gap-x-4 gap-y-8 sm:grid-cols-3 xl:grid-cols-4">{cells}</div>;
}
