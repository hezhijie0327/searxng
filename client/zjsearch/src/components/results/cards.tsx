// SPDX-License-Identifier: AGPL-3.0-or-later

import { type ReactNode, useState } from "react";
import { formatDate, formatLength } from "../../lib/format.ts";
import { useT } from "../../lib/i18n.ts";
import type { GlobalData, ResultItem } from "../../lib/types.ts";
import {
  CalendarIcon,
  ClockIcon,
  CloseIcon,
  DownloadIcon,
  ExternalLinkIcon,
  EyeIcon,
  FileIcon,
  FilmIcon,
  MagnetIcon,
  MusicIcon,
  PackageIcon,
  PlayIcon,
} from "../icons.tsx";
import { MapResult } from "./MapView.tsx";

export const THEME_STATIC = "/static/themes/zjsearch";

// ------------------------------------------------------------- shared parts

function ResultLink({
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
  const newTab = globals.results_on_new_tab;
  return (
    <a
      className={className}
      href={url}
      {...(newTab ? { target: "_blank", rel: "noopener noreferrer" } : { rel: "noreferrer" })}
    >
      {children}
    </a>
  );
}

function Favicon({ result }: { result: ResultItem }) {
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

function PrettyUrl({ result, globals }: { result: ResultItem; globals: GlobalData }) {
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

function Title({ result, globals }: { result: ResultItem; globals: GlobalData }) {
  return (
    <h3 className="text-base font-medium leading-snug">
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

function MetaLine({ result }: { result: ResultItem }) {
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

function EnginesLine({ result }: { result: ResultItem }) {
  const shown = result.engines.slice(0, MAX_ENGINES_SHOWN);
  const hidden = result.engines.length - shown.length;
  return (
    <div className="mt-2 flex min-w-0 flex-wrap items-center gap-x-2 gap-y-1 text-xs text-ink-3">
      {shown.map((engine) => (
        <span className="rounded-full bg-surface-2 px-2 py-0.5" key={engine}>
          {engine}
        </span>
      ))}
      {hidden > 0 ? (
        <span className="rounded-full bg-surface-2 px-2 py-0.5" title={result.engines.join(", ")}>
          +{hidden}
        </span>
      ) : null}
    </div>
  );
}

function Thumb({
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

function ResultArticle({ children, priority, id }: { children: ReactNode; priority?: string; id?: string }) {
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

function MediaCollapse({
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

function EmbedFrame({ src }: { src: string }) {
  return (
    <div className="aspect-video w-full max-w-3xl overflow-hidden rounded-xl border border-line bg-black">
      <iframe allowFullScreen className="size-full" referrerPolicy="origin" src={src} title="embedded content" />
    </div>
  );
}

/** Always-visible preview (music intent): prefer our own audio player for
    stream URLs; when the source is not raw audio fall back to the embed. */
function MediaPreview({ src, video = false }: { src: string; video?: boolean }) {
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

interface CardProps {
  result: ResultItem;
  globals: GlobalData;
  /** map-intent pages open the inline OSM map automatically (upstream simple behaviour) */
  autoOpenMap?: boolean;
  /** music-intent pages show the media preview expanded with our own player */
  mediaOpen?: boolean;
  /** first results load their thumbnail eagerly (LCP) */
  eager?: boolean;
}

export function DefaultCard({ eager, result, globals, mediaOpen }: CardProps) {
  const t = useT();
  return (
    <ResultArticle priority={result.priority}>
      <div className="flex gap-4">
        <div className="min-w-0 flex-1">
          <PrettyUrl globals={globals} result={result} />
          <div className="mt-1">
            <Title globals={globals} result={result} />
          </div>
          <div className="mt-1">
            <MetaLine result={result} />
          </div>
          {result.iframe_src ? (
            <div className="mt-2">
              {mediaOpen ? (
                <MediaPreview src={result.iframe_src} />
              ) : (
                <MediaCollapse hideLabel={t("hide_media")} showLabel={t("show_media")}>
                  {() => <EmbedFrame src={result.iframe_src ?? ""} />}
                </MediaCollapse>
              )}
            </div>
          ) : null}
          <p
            className="mt-1.5 line-clamp-3 text-sm leading-relaxed text-ink-2"
            dangerouslySetInnerHTML={{
              __html: result.content_html || t("no_description"),
            }}
            dir="auto"
          />
          {result.audio_src ? (
            <audio className="mt-2 w-full max-w-md" controls preload="none" src={result.audio_src} />
          ) : null}
        </div>
        {result.thumbnail ? (
          <ResultLink className="shrink-0 self-start" globals={globals} result={result}>
            <Thumb
              alt={result.title_text}
              className="h-24 w-40"
              eager={eager}
              lengthDisplay={formatLength(result.length_display, result.length_seconds)}
              src={result.thumbnail}
            />
          </ResultLink>
        ) : null}
      </div>
      <EnginesLine result={result} />
    </ResultArticle>
  );
}

export function VideoCard({ eager, result, globals, mediaOpen }: CardProps) {
  const t = useT();
  const [previewOpen, setPreviewOpen] = useState(false);
  const hasMedia = Boolean(result.iframe_src);
  return (
    <ResultArticle priority={result.priority}>
      <div className="flex gap-4">
        <div className="min-w-0 flex-1">
          <PrettyUrl globals={globals} result={result} />
          <div className="mt-1">
            <Title globals={globals} result={result} />
          </div>
          <div className="mt-1">
            <MetaLine result={result} />
          </div>
          {hasMedia && previewOpen ? (
            <div className="mt-2 animate-fade-in">
              <MediaPreview src={result.iframe_src ?? ""} video={!mediaOpen} />
            </div>
          ) : null}
          <p
            className="mt-1.5 line-clamp-3 text-sm leading-relaxed text-ink-2"
            dangerouslySetInnerHTML={{ __html: result.content_html || t("no_description") }}
            dir="auto"
          />
        </div>
        {result.thumbnail ? (
          <div className="relative shrink-0">
            <ResultLink className="block" globals={globals} result={result}>
              <Thumb
                alt={result.title_text}
                className="h-24 w-40"
                eager={eager}
                lengthDisplay={formatLength(result.length_display, result.length_seconds)}
                src={result.thumbnail}
              />
            </ResultLink>
            {hasMedia ? (
              <button
                aria-label={previewOpen ? t("hide_video") : t("show_video")}
                className="absolute end-1 top-1 z-10 grid size-7 place-items-center rounded-full bg-black/70 text-white transition-colors hover:bg-accent-strong hover:text-ink"
                onClick={() => {
                  setPreviewOpen((value) => !value);
                }}
                title={previewOpen ? t("hide_video") : t("show_video")}
                type="button"
              >
                {previewOpen ? <CloseIcon className="size-3.5" /> : <PlayIcon className="size-3.5" />}
              </button>
            ) : null}
          </div>
        ) : null}
      </div>
      <EnginesLine result={result} />
    </ResultArticle>
  );
}

/** News-intent layout: same link style as every card, compact snippet, 16:9 thumb. */
export function NewsCard({ result, globals }: CardProps) {
  return (
    <ResultArticle priority={result.priority}>
      <PrettyUrl globals={globals} result={result} />
      <div className="mt-1">
        <Title globals={globals} result={result} />
      </div>
      <div className="mt-1">
        <MetaLine result={result} />
      </div>
      <p
        className="mt-1 line-clamp-2 text-sm leading-relaxed text-ink-2"
        dangerouslySetInnerHTML={{ __html: result.content_html }}
        dir="auto"
      />
      <EnginesLine result={result} />
    </ResultArticle>
  );
}

export function TorrentCard({ result, globals }: CardProps) {
  const t = useT();
  return (
    <ResultArticle priority={result.priority}>
      <PrettyUrl globals={globals} result={result} />
      <div className="mt-1">
        <Title globals={globals} result={result} />
      </div>
      <div className="mt-1">
        <MetaLine result={result} />
      </div>
      <div className="mt-2 flex flex-wrap items-center gap-2 text-xs">
        {result.magnetlink ? (
          <a
            className="inline-flex items-center gap-1.5 rounded-full bg-surface-2 px-3 py-1 text-ink-2 transition-colors hover:text-ink"
            href={result.magnetlink}
          >
            <MagnetIcon className="size-3.5" />
            {t("magnet_link")}
          </a>
        ) : null}
        {result.torrentfile ? (
          <a
            className="inline-flex items-center gap-1.5 rounded-full bg-surface-2 px-3 py-1 text-ink-2 transition-colors hover:text-ink"
            href={result.torrentfile}
          >
            <DownloadIcon className="size-3.5" />
            {t("torrent_file")}
          </a>
        ) : null}
        {result.seed !== undefined ? (
          <span className="inline-flex items-center gap-1 rounded-full bg-surface-2 px-3 py-1 text-ink-2">
            <span className="size-1.5 rounded-full bg-ok" />
            {result.seed} {t("seeder")}
          </span>
        ) : null}
        {result.leech !== undefined ? (
          <span className="inline-flex items-center gap-1 rounded-full bg-surface-2 px-3 py-1 text-ink-2">
            <span className="size-1.5 rounded-full bg-danger" />
            {result.leech} {t("leecher")}
          </span>
        ) : null}
        {result.filesize ? (
          <span className="inline-flex items-center gap-1 rounded-full bg-surface-2 px-3 py-1 text-ink-2">
            <FileIcon className="size-3.5" />
            {result.filesize}
          </span>
        ) : null}
        {result.files ? (
          <span className="inline-flex items-center gap-1 rounded-full bg-surface-2 px-3 py-1 text-ink-2">
            {result.files} {t("files")}
          </span>
        ) : null}
      </div>
      {result.content_html ? (
        <p
          className="mt-2 line-clamp-2 text-sm text-ink-2"
          dangerouslySetInnerHTML={{ __html: result.content_html }}
          dir="auto"
        />
      ) : null}
      <EnginesLine result={result} />
    </ResultArticle>
  );
}

export function ProductCard({ result, globals }: CardProps) {
  return (
    <ResultArticle priority={result.priority}>
      <div className="flex gap-4">
        <div className="min-w-0 flex-1">
          <PrettyUrl globals={globals} result={result} />
          <div className="mt-1">
            <Title globals={globals} result={result} />
          </div>
          <div className="mt-1">
            <MetaLine result={result} />
          </div>
          <div className="mt-2 flex flex-wrap items-baseline gap-x-3 gap-y-1">
            {result.price ? <span className="text-lg font-semibold text-ink">{result.price}</span> : null}
            {result.shipping ? <span className="text-xs text-ink-3">{result.shipping}</span> : null}
            {result.source_country ? <span className="text-xs text-ink-3">{result.source_country}</span> : null}
          </div>
          {result.content_html ? (
            <p
              className="mt-1.5 line-clamp-3 text-sm text-ink-2"
              dangerouslySetInnerHTML={{ __html: result.content_html }}
              dir="auto"
            />
          ) : null}
        </div>
        {result.thumbnail ? (
          <ResultLink className="shrink-0 self-start" globals={globals} result={result}>
            <Thumb alt={result.title_text} className="h-28 w-28" src={result.thumbnail} />
          </ResultLink>
        ) : null}
      </div>
      <EnginesLine result={result} />
    </ResultArticle>
  );
}

/** Kagi-shopping style tiles for product-only result pages. */
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

export function KeyValueCard({ result }: CardProps) {
  return (
    <ResultArticle priority={result.priority}>
      <div className="overflow-hidden rounded-xl border border-line">
        <table className="w-full text-sm">
          {result.caption ? (
            <caption className="bg-surface-2 px-4 py-2 text-left font-medium">{result.caption}</caption>
          ) : null}
          {result.key_title || result.value_title ? (
            <thead>
              <tr className="bg-surface-2 text-xs text-ink-2">
                <th className="px-4 py-2 text-left font-medium" scope="col">
                  {result.key_title}
                </th>
                <th className="px-4 py-2 text-left font-medium" scope="col">
                  {result.value_title}
                </th>
              </tr>
            </thead>
          ) : null}
          <tbody>
            {Object.entries(result.kvmap ?? {}).map(([key, value], index) => (
              <tr className={index % 2 === 0 ? "bg-surface" : "bg-bg/60"} key={key}>
                <th className="px-4 py-1.5 text-left font-medium text-ink-2" scope="row">
                  {key}
                </th>
                <td className="px-4 py-1.5 text-ink">{String(value)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <EnginesLine result={result} />
    </ResultArticle>
  );
}

export function CodeCard({ result, globals }: CardProps) {
  const t = useT();
  return (
    <ResultArticle priority={result.priority}>
      <PrettyUrl globals={globals} result={result} />
      <div className="mt-1 flex flex-wrap items-baseline gap-x-3">
        <Title globals={globals} result={result} />
        {result.filename ? (
          <span className="text-xs text-ink-3">
            {t("filename")}: <code className="font-mono">{result.filename}</code>
          </span>
        ) : null}
      </div>
      {result.repository ? (
        <p className="mt-1 text-xs text-ink-3">
          {t("repository")}:{" "}
          <a className="text-accent hover:underline" href={result.repository} rel="noreferrer" target="_blank">
            {result.repository}
          </a>
        </p>
      ) : null}
      {result.content_html ? (
        <p className="mt-1.5 text-sm text-ink-2" dangerouslySetInnerHTML={{ __html: result.content_html }} dir="auto" />
      ) : null}
      {result.code_html ? (
        <pre
          className="mt-2 max-h-96 overflow-auto rounded-xl border border-line bg-surface-2 p-4 font-mono text-xs leading-relaxed"
          dangerouslySetInnerHTML={{ __html: result.code_html }}
          dir="ltr"
        />
      ) : null}
      <EnginesLine result={result} />
    </ResultArticle>
  );
}

export function FileCard({ result, globals }: CardProps) {
  const t = useT();
  const attributes: Array<[string, string | undefined]> = [
    [t("author"), result.author],
    [t("filename"), result.filename],
    [t("filesize"), result.size],
    [t("date"), result.time],
    [t("type"), result.mimetype],
  ];
  const isMedia = result.mtype === "audio" || result.mtype === "video";
  return (
    <ResultArticle priority={result.priority}>
      <PrettyUrl globals={globals} result={result} />
      <div className="mt-1 flex items-baseline gap-2">
        <FileIcon className="size-4 shrink-0 text-ink-3" />
        <Title globals={globals} result={result} />
      </div>
      <div className="mt-1">
        <MetaLine result={result} />
      </div>
      {result.abstract_html ? (
        <p
          className="mt-1.5 text-sm text-ink-2"
          dangerouslySetInnerHTML={{ __html: result.abstract_html }}
          dir="auto"
        />
      ) : null}
      {result.content_html ? (
        <p className="mt-1 text-sm text-ink-2" dangerouslySetInnerHTML={{ __html: result.content_html }} dir="auto" />
      ) : null}
      <dl className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-xs text-ink-3">
        {attributes
          .filter(([, value]) => Boolean(value))
          .map(([label, value]) => (
            <div className="flex gap-1" key={label}>
              <dt>{label}:</dt>
              <dd className="text-ink-2">{value}</dd>
            </div>
          ))}
      </dl>
      {result.embedded ? (
        isMedia ? (
          result.mtype === "video" ? (
            <MediaCollapse hideLabel={t("hide_media")} showLabel={t("show_media")}>
              {() => (
                <video
                  className="w-full max-w-lg rounded-xl"
                  controls
                  poster={result.thumbnail}
                  preload="metadata"
                  src={result.embedded}
                />
              )}
            </MediaCollapse>
          ) : (
            // audio: inline player, no collapse - music results should be
            // playable in one click (preload="none" keeps it cheap)
            <audio className="mt-2 w-full max-w-md" controls preload="none" src={result.embedded} />
          )
        ) : (
          <a
            className="mt-2 inline-flex items-center gap-1.5 rounded-full bg-surface-2 px-3 py-1 text-xs text-ink-2 hover:text-ink"
            download
            href={result.embedded}
            rel="noreferrer"
            target="_blank"
          >
            <DownloadIcon className="size-3.5" />
            {t("download")}
          </a>
        )
      ) : null}
      <EnginesLine result={result} />
    </ResultArticle>
  );
}

export function PaperCard({ result, globals }: CardProps) {
  const t = useT();
  const details: Array<[string, ReactNode]> = [];
  if (result.authors && result.authors.length > 0) {
    details.push([t("author"), result.authors.join(", ")]);
  }
  if (result.published_date) {
    details.push([t("date"), formatDate(result.published_date)]);
  }
  if (result.journal) {
    details.push([
      "",
      <>
        {result.journal}
        {result.volume ? `, ${result.volume}` : ""}
        {result.number ? ` (${result.number})` : ""}
        {result.pages ? `, ${result.pages}` : ""}
      </>,
    ]);
  }
  if (result.editor) {
    details.push(["Editor", result.editor]);
  }
  if (result.publisher) {
    details.push(["Publisher", result.publisher]);
  }
  if (result.paper_type) {
    details.push([t("type"), result.paper_type]);
  }
  if (result.doi) {
    details.push([
      "DOI",
      <a
        className="text-accent hover:underline"
        href={`https://${globals.doi_resolver}/${result.doi}`}
        key="doi"
        rel="noreferrer"
        target="_blank"
      >
        {result.doi}
      </a>,
    ]);
  }
  if (result.issn && result.issn.length > 0) {
    details.push(["ISSN", result.issn.join(", ")]);
  }
  if (result.isbn && result.isbn.length > 0) {
    details.push(["ISBN", result.isbn.join(", ")]);
  }
  return (
    <ResultArticle priority={result.priority}>
      <PrettyUrl globals={globals} result={result} />
      <div className="mt-1">
        <Title globals={globals} result={result} />
      </div>
      <div className="mt-1">
        <MetaLine result={result} />
      </div>
      {result.content_html ? (
        <p
          className="mt-1.5 line-clamp-4 text-sm leading-relaxed text-ink-2"
          dangerouslySetInnerHTML={{ __html: result.content_html }}
          dir="auto"
        />
      ) : null}
      {result.comments ? (
        <p className="mt-1.5 text-sm italic text-ink-3" dir="auto">
          {result.comments}
        </p>
      ) : null}
      <dl className="mt-2 grid grid-cols-[auto_1fr] gap-x-3 gap-y-1 text-xs">
        {details.map(([label, value], index) => (
          <div className="col-span-2 grid grid-cols-subgrid items-baseline" key={index}>
            {label ? <dt className="text-ink-3">{label}:</dt> : <dt />}
            <dd className="text-ink-2">{value}</dd>
          </div>
        ))}
      </dl>
      <div className="mt-2 flex flex-wrap gap-2 text-xs">
        {result.pdf_url ? (
          <a
            className="inline-flex items-center gap-1.5 rounded-full bg-accent-soft px-3 py-1 font-medium text-accent transition-colors hover:bg-accent-strong hover:text-accent-contrast"
            href={result.pdf_url}
            rel="noreferrer"
            target="_blank"
          >
            <FileIcon className="size-3.5" />
            PDF
          </a>
        ) : null}
        {result.html_url ? (
          <a
            className="inline-flex items-center gap-1.5 rounded-full bg-surface-2 px-3 py-1 text-ink-2 hover:text-ink"
            href={result.html_url}
            rel="noreferrer"
            target="_blank"
          >
            <ExternalLinkIcon className="size-3.5" />
            HTML
          </a>
        ) : null}
        {result.doi ? (
          <a
            className="inline-flex items-center gap-1.5 rounded-full bg-surface-2 px-3 py-1 text-ink-2 hover:text-ink"
            href={`https://www.altmetric.com/details/doi/${result.doi}`}
            rel="noreferrer"
            target="_blank"
          >
            Altmetric
          </a>
        ) : null}
      </div>
      {result.tags && result.tags.length > 0 ? (
        <div className="mt-2 flex flex-wrap gap-1.5">
          {result.tags.map((tag) => (
            <span className="rounded-full bg-surface-2 px-2 py-0.5 text-xs text-ink-3" key={tag}>
              {tag}
            </span>
          ))}
        </div>
      ) : null}
      <EnginesLine result={result} />
    </ResultArticle>
  );
}

export function PackageCard({ result, globals }: CardProps) {
  const t = useT();
  return (
    <ResultArticle priority={result.priority}>
      <PrettyUrl globals={globals} result={result} />
      <div className="mt-1 flex flex-wrap items-baseline gap-x-3 gap-y-1">
        <Title globals={globals} result={result} />
        {result.version ? (
          <code className="rounded bg-surface-2 px-1.5 py-0.5 text-xs text-ink-2">{result.version}</code>
        ) : null}
        {result.package_name && result.package_name !== result.title_text ? (
          <code className="text-xs text-ink-2">{result.package_name}</code>
        ) : null}
      </div>
      {result.content_html ? (
        <p
          className="mt-1.5 line-clamp-3 text-sm text-ink-2"
          dangerouslySetInnerHTML={{ __html: result.content_html }}
          dir="auto"
        />
      ) : null}
      <dl className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-xs text-ink-3">
        {result.maintainer ? (
          <div className="flex gap-1">
            <dt>{t("author")}:</dt>
            <dd className="text-ink-2">{result.maintainer}</dd>
          </div>
        ) : null}
        {result.published_date ? (
          <div className="flex gap-1">
            <dt>Updated:</dt>
            <dd className="text-ink-2">{formatDate(result.published_date)}</dd>
          </div>
        ) : null}
        {result.popularity ? (
          <div className="flex gap-1">
            <dt>Popularity:</dt>
            <dd className="text-ink-2">{result.popularity}</dd>
          </div>
        ) : null}
        {result.license_name ? (
          <div className="flex gap-1">
            <dt>License:</dt>
            <dd className="text-ink-2">
              {result.license_url ? (
                <a className="text-accent hover:underline" href={result.license_url} rel="noreferrer" target="_blank">
                  {result.license_name}
                </a>
              ) : (
                result.license_name
              )}
            </dd>
          </div>
        ) : null}
      </dl>
      <div className="mt-2 flex flex-wrap gap-2 text-xs">
        {result.homepage ? (
          <a
            className="inline-flex items-center gap-1.5 rounded-full bg-accent-soft px-3 py-1 font-medium text-accent transition-colors hover:bg-accent-strong hover:text-accent-contrast"
            href={result.homepage}
            rel="noreferrer"
            target="_blank"
          >
            <ExternalLinkIcon className="size-3.5" />
            Homepage
          </a>
        ) : null}
        {result.source_code_url ? (
          <a
            className="inline-flex items-center gap-1.5 rounded-full bg-surface-2 px-3 py-1 text-ink-2 hover:text-ink"
            href={result.source_code_url}
            rel="noreferrer"
            target="_blank"
          >
            <PackageIcon className="size-3.5" />
            Source code
          </a>
        ) : null}
        {Object.entries(result.project_links ?? {}).map(([name, url]) => (
          <a
            className="inline-flex items-center gap-1.5 rounded-full bg-surface-2 px-3 py-1 text-ink-2 hover:text-ink"
            href={url}
            key={url}
            rel="noreferrer"
            target="_blank"
          >
            <ExternalLinkIcon className="size-3.5" />
            {name}
          </a>
        ))}
      </div>
      {result.tags && result.tags.length > 0 ? (
        <div className="mt-2 flex flex-wrap gap-1.5">
          {result.tags.map((tag) => (
            <span className="rounded-full bg-surface-2 px-2 py-0.5 text-xs text-ink-3" key={tag}>
              {tag}
            </span>
          ))}
        </div>
      ) : null}
      <EnginesLine result={result} />
    </ResultArticle>
  );
}

export function MapCard({ result, globals, autoOpenMap }: CardProps) {
  const t = useT();
  const address = result.address;
  const addressLine = address
    ? [address.name, address.road, address.house_number, address.postcode, address.locality, address.country]
        .filter(Boolean)
        .join(", ")
    : "";
  return (
    <ResultArticle priority={result.priority}>
      <PrettyUrl globals={globals} result={result} />
      <div className="mt-1">
        <Title globals={globals} result={result} />
      </div>
      <div className="mt-1">
        <MetaLine result={result} />
      </div>
      {result.content_html ? (
        <p className="mt-1.5 text-sm text-ink-2" dangerouslySetInnerHTML={{ __html: result.content_html }} dir="auto" />
      ) : null}
      {addressLine ? (
        <p className="mt-2 text-sm text-ink-2">
          <span className="text-ink-3">{t("address")}: </span>
          {addressLine}
        </p>
      ) : null}
      {result.data && result.data.length > 0 ? (
        <dl className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-xs text-ink-3">
          {result.data.map((item) => (
            <div className="flex gap-1" key={item.label}>
              <dt>{item.label}:</dt>
              <dd className="text-ink-2">{item.value}</dd>
            </div>
          ))}
        </dl>
      ) : null}
      <MapResult
        autoOpen={autoOpenMap}
        boundingbox={result.boundingbox}
        geojson={result.geojson}
        label={t("show_map")}
        latitude={result.latitude}
        longitude={result.longitude}
      />
      {result.map_links && result.map_links.length > 0 ? (
        <div className="mt-2 flex flex-wrap gap-2 text-xs">
          {result.map_links.map((link) => (
            <a
              className="inline-flex items-center gap-1.5 rounded-full bg-surface-2 px-3 py-1 text-ink-2 hover:text-ink"
              href={link.url}
              key={link.url}
              rel="noreferrer"
              target="_blank"
            >
              <ExternalLinkIcon className="size-3.5" />
              {link.label}
            </a>
          ))}
        </div>
      ) : null}
      <EnginesLine result={result} />
    </ResultArticle>
  );
}

export function ImageListCard({ result, globals, onOpen }: CardProps & { onOpen: () => void }) {
  const thumbSrc = result.thumbnail_src || result.img_src || "";
  return (
    <ResultArticle priority={result.priority}>
      <div className="flex gap-4">
        <button
          aria-label={result.title_text}
          className="group/img relative size-24 shrink-0 overflow-hidden rounded-xl bg-surface-2"
          onClick={onOpen}
          type="button"
        >
          <img
            alt={result.title_text}
            className="size-full object-cover transition-transform duration-300 group-hover:scale-[1.04]"
            decoding="async"
            loading="lazy"
            onError={(event) => {
              event.currentTarget.src = `${THEME_STATIC}/img/img_load_error.svg`;
            }}
            src={thumbSrc}
          />
          <span className="absolute inset-0 grid place-items-center bg-black/0 text-white opacity-0 transition-all group-hover/img:bg-black/30 group-hover/img:opacity-100">
            <FilmIcon className="size-5" />
          </span>
        </button>
        <div className="min-w-0 flex-1">
          <PrettyUrl globals={globals} result={result} />
          <div className="mt-1">
            <Title globals={globals} result={result} />
          </div>
          <div className="mt-1 flex flex-wrap gap-x-3 text-xs text-ink-3">
            {result.resolution ? <span>{result.resolution}</span> : null}
            {result.img_format ? <span>{result.img_format}</span> : null}
            {result.filesize ? <span>{result.filesize}</span> : null}
            {result.source ? <span>{result.source}</span> : null}
          </div>
          <EnginesLine result={result} />
        </div>
      </div>
    </ResultArticle>
  );
}

/** Kagi-style video tiles for video-only result pages.  Tiles with an
    embeddable source get a Spotify-style hover play button that expands the
    player in place. */
export function VideoGrid({ results, globals }: { results: ResultItem[]; globals: GlobalData }) {
  const t = useT();
  const [playing, setPlaying] = useState<number | null>(null);
  return (
    <div className="grid grid-cols-1 gap-x-4 gap-y-8 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
      {results.map((result, index) => {
        const length = formatLength(result.length_display, result.length_seconds);
        const isPlaying = playing === index;
        return (
          <article className="group" key={`${result.url}-${index}`}>
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
                {length ? (
                  <span className="absolute bottom-2 right-2 rounded bg-black/80 px-1.5 py-0.5 text-[11px] font-medium text-white">
                    {length}
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
              {result.iframe_src && isPlaying ? (
                <div className="absolute inset-0 animate-fade-in overflow-hidden rounded-xl border border-line bg-black">
                  <iframe
                    allowFullScreen
                    className="size-full"
                    referrerPolicy="origin"
                    src={result.iframe_src ?? ""}
                    title={result.title_text}
                  />
                </div>
              ) : null}
              {result.iframe_src ? (
                <button
                  aria-label={isPlaying ? t("hide_video") : t("show_video")}
                  className="absolute end-2 top-2 z-10 grid size-7 place-items-center rounded-full bg-black/70 text-white transition-colors hover:bg-accent-strong"
                  onClick={() => {
                    setPlaying(isPlaying ? null : index);
                  }}
                  title={isPlaying ? t("hide_video") : t("show_video")}
                  type="button"
                >
                  {isPlaying ? <CloseIcon className="size-3.5" /> : <EyeIcon className="size-3.5" />}
                </button>
              ) : null}
            </div>
            <h3 className="mt-2.5 line-clamp-2 text-base font-medium leading-snug">
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
              <span className="shrink-0">{result.published_date ? formatDate(result.published_date) : null}</span>
            </div>
            {result.views ? <div className="mt-0.5 text-xs text-ink-3">{result.views}</div> : null}
          </article>
        );
      })}
    </div>
  );
}

export function FallbackCard(props: CardProps) {
  return <DefaultCard {...props} />;
}

export function ResultCard(props: CardProps & { onOpenImage?: () => void }) {
  const { result } = props;
  switch (result.template) {
    case "images":
      return props.onOpenImage ? <ImageListCard {...props} onOpen={props.onOpenImage} /> : <DefaultCard {...props} />;
    case "videos":
      return <VideoCard {...props} />;
    case "news":
      return <NewsCard globals={props.globals} result={props.result} />;
    case "torrent":
      return <TorrentCard {...props} />;
    case "map":
      return <MapCard {...props} />;
    case "paper":
      return <PaperCard {...props} />;
    case "packages":
      return <PackageCard {...props} />;
    case "code":
      return <CodeCard {...props} />;
    case "file":
      return <FileCard {...props} />;
    case "keyvalue":
      return <KeyValueCard {...props} />;
    case "products":
      return <ProductCard {...props} />;
    default:
      return <DefaultCard {...props} />;
  }
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
