// SPDX-License-Identifier: AGPL-3.0-or-later

import { type ReactNode, useState } from "react";
import { THEME_STATIC } from "../../lib/constants.ts";
import { formatDate, formatLength } from "../../lib/format.ts";
import { useT } from "../../lib/i18n.ts";
import { newTabLinkProps } from "../../lib/link.ts";
import type { GlobalData, ResultItem } from "../../lib/types.ts";
import { CalendarIcon, ClockIcon, MusicIcon, PlayIcon } from "../icons.tsx";

// ------------------------------------------------------------- shared parts

export function ResultLink({
  result,
  globals,
  href,
  className,
  children,
}: {
  result: ResultItem;
  globals: GlobalData;
  href?: string;
  className?: string;
  children: ReactNode;
}) {
  const url = href ?? result.url;
  return (
    <a className={className} href={url} {...newTabLinkProps(globals.results_on_new_tab)}>
      {children}
    </a>
  );
}

export function Favicon({ result }: { result: ResultItem }) {
  if (!result.favicon) {
    return null;
  }
  return (
    <img
      alt=""
      className="size-4 shrink-0 rounded-sm object-contain"
      decoding="async"
      loading="lazy"
      onError={(event) => {
        event.currentTarget.src = `${THEME_STATIC}/img/empty_favicon.svg`;
      }}
      src={result.favicon}
    />
  );
}

export function PrettyUrl({ result, globals }: { result: ResultItem; globals: GlobalData }) {
  if (!result.pretty_url || result.pretty_url.length === 0) {
    return null;
  }
  return (
    <ResultLink
      className="flex min-w-0 items-center gap-1.5 text-xs text-ink-3 group-hover:text-ink-2"
      globals={globals}
      result={result}
    >
      <Favicon result={result} />
      <span className="truncate" dir="ltr">
        {result.pretty_url.join("")}
      </span>
    </ResultLink>
  );
}

export function Title({ result, globals }: { result: ResultItem; globals: GlobalData }) {
  return (
    <h3 className="line-clamp-1 text-base font-medium leading-snug">
      <ResultLink
        className="text-ink decoration-accent/50 underline-offset-2 hover:text-accent hover:underline"
        globals={globals}
        result={result}
      >
        <span dangerouslySetInnerHTML={{ __html: result.title_html }} dir="auto" />
      </ResultLink>
    </h3>
  );
}

export function MetaLine({ result }: { result: ResultItem }) {
  const t = useT();
  const bits: ReactNode[] = [];
  if (result.published_date) {
    bits.push(
      <span className="inline-flex items-center gap-1" key="date">
        <CalendarIcon className="size-3" />
        {formatDate(result.published_date)}
      </span>,
    );
  }
  if (result.author) {
    bits.push(
      <span className="truncate" key="author">
        {result.author}
      </span>,
    );
  }
  if (result.views) {
    bits.push(<span key="views">{result.views}</span>);
  }
  const length = formatLength(result.length_display, result.length_seconds);
  if (length) {
    bits.push(
      <span className="inline-flex items-center gap-1" key="length">
        <ClockIcon className="size-3" />
        {length}
      </span>,
    );
  }
  if (bits.length === 0) {
    return null;
  }
  return (
    <div className="flex flex-wrap items-center gap-x-3 gap-y-0.5 text-xs text-ink-3">
      {bits}
      {result.metadata ? (
        <span className="rounded bg-accent-soft px-1.5 py-0.5 text-accent" dir="auto">
          {result.metadata}
        </span>
      ) : null}
      <span className="sr-only">{t("response_time")}</span>
    </div>
  );
}

const MAX_ENGINES_SHOWN = 3;

export function EnginesLine({ result, leading }: { result: ResultItem; leading?: ReactNode }) {
  const t = useT();
  const [expanded, setExpanded] = useState(false);
  const engines = result.engines;
  const shown = expanded ? engines : engines.slice(0, MAX_ENGINES_SHOWN);
  const hidden = engines.length - MAX_ENGINES_SHOWN;
  return (
    <div className="mt-2 flex min-w-0 flex-wrap items-center gap-x-2 gap-y-1 text-xs text-ink-3">
      {leading}
      {shown.map((engine) => (
        <span className="rounded-full bg-surface-2 px-2 py-0.5" key={engine}>
          {engine}
        </span>
      ))}
      {!expanded && hidden > 0 ? (
        <button
          className="rounded-full bg-surface-2 px-2 py-0.5 transition-colors hover:text-ink"
          onClick={() => {
            setExpanded(true);
          }}
          title={engines.join(", ")}
          type="button"
        >
          +{hidden}
        </button>
      ) : null}
      {expanded && hidden > 0 ? (
        <button
          className="transition-colors hover:text-ink"
          onClick={() => {
            setExpanded(false);
          }}
          type="button"
        >
          {t("show_less")}
        </button>
      ) : null}
    </div>
  );
}

export function Thumb({
  src,
  alt,
  className,
  lengthDisplay,
  eager,
}: {
  src: string;
  alt: string;
  className?: string;
  lengthDisplay?: string | null;
  eager?: boolean;
}) {
  return (
    <div className={`relative shrink-0 overflow-hidden rounded-xl bg-surface-2 ${className ?? ""}`}>
      <img
        alt={alt}
        className="size-full object-cover transition-transform duration-300 group-hover:scale-[1.04]"
        decoding="async"
        fetchPriority={eager ? "high" : undefined}
        loading={eager ? "eager" : "lazy"}
        onError={(event) => {
          event.currentTarget.src = `${THEME_STATIC}/img/img_load_error.svg`;
        }}
        src={src}
      />
      {lengthDisplay ? (
        <span className="absolute bottom-1 right-1 rounded bg-black/70 px-1.5 py-0.5 text-[11px] font-medium text-white">
          {lengthDisplay}
        </span>
      ) : null}
    </div>
  );
}

export function ResultArticle({ children, priority, id }: { children: ReactNode; priority?: string; id?: string }) {
  return (
    <article
      className={`group relative scroll-mt-32 rounded-2xl border border-transparent p-4 transition-colors hover:bg-surface ${
        priority === "low" ? "opacity-70" : ""
      }`}
      data-priority={priority || undefined}
      id={id}
    >
      {children}
    </article>
  );
}

export function MediaCollapse({
  showLabel,
  hideLabel,
  children,
}: {
  showLabel: string;
  hideLabel: string;
  children: (open: boolean) => ReactNode;
}) {
  const [open, setOpen] = useState(false);
  return (
    <div>
      <button
        className="mt-1 inline-flex items-center gap-1.5 rounded-full bg-surface-2 px-3 py-1 text-xs text-ink-2 transition-colors hover:text-ink"
        onClick={() => {
          setOpen((prev) => !prev);
        }}
        type="button"
      >
        <PlayIcon className="size-3" />
        {open ? hideLabel : showLabel}
      </button>
      {open ? <div className="mt-2 animate-fade-in">{children(open)}</div> : null}
    </div>
  );
}

export function EmbedFrame({ src }: { src: string }) {
  return (
    <div className="aspect-video w-full max-w-3xl overflow-hidden rounded-xl border border-line bg-black">
      <iframe allowFullScreen className="size-full" referrerPolicy="origin" src={src} title="embedded content" />
    </div>
  );
}

/** Always-visible preview (music intent): prefer our own audio player for
    stream URLs; when the source is not raw audio fall back to the embed. */
export function MediaPreview({ src, video = false }: { src: string; video?: boolean }) {
  const [audioFailed, setAudioFailed] = useState(false);
  if (video || audioFailed) {
    return <EmbedFrame src={src} />;
  }
  return (
    <div className="flex max-w-3xl items-center gap-3 rounded-2xl border border-line bg-surface px-4 py-3">
      <span className="grid size-9 shrink-0 place-items-center rounded-full bg-accent-soft text-accent">
        <MusicIcon className="size-4" />
      </span>
      <audio
        className="h-9 w-full"
        controls
        onError={() => {
          setAudioFailed(true);
        }}
        preload="none"
        src={src}
      />
    </div>
  );
}

// -------------------------------------------------------------- card shells

export interface CardProps {
  result: ResultItem;
  globals: GlobalData;
  /** map-intent pages open the inline OSM map automatically (upstream simple behaviour) */
  autoOpenMap?: boolean;
  /** music-intent pages show the media preview expanded with our own player */
  mediaOpen?: boolean;
  /** first results load their thumbnail eagerly (LCP) */
  eager?: boolean;
}
export function ResultSkeleton() {
  return (
    <div className="rounded-2xl p-4">
      <div className="zjs-skeleton h-3 w-40" />
      <div className="zjs-skeleton mt-3 h-5 w-3/4" />
      <div className="zjs-skeleton mt-3 h-3 w-full" />
      <div className="zjs-skeleton mt-2 h-3 w-5/6" />
      <div className="zjs-skeleton mt-4 h-3 w-24" />
    </div>
  );
}
