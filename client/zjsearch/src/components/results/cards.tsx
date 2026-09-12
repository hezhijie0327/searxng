// SPDX-License-Identifier: Apache-2.0 WITH Commons-Clause-1.0

// SPDX-License-Identifier: Apache-2.0 WITH Commons-Clause-1.0

import { memo, type ReactNode, useState } from "react";
import { formatDate, formatLength } from "../../lib/format.ts";
import { useT } from "../../lib/i18n.ts";
import {
  ArrowDownIcon,
  ArrowUpIcon,
  CalendarIcon,
  CloseIcon,
  CodeIcon,
  DownloadIcon,
  ExternalLinkIcon,
  FileIcon,
  FilmIcon,
  MagnetIcon,
  MusicIcon,
  PackageIcon,
  PlayIcon,
  StarIcon,
} from "../icons.tsx";

import {
  type CardProps,
  EmbedFrame,
  EnginesLine,
  MediaCollapse,
  MediaPreview,
  MetaLine,
  PrettyUrl,
  ResultArticle,
  ResultLink,
  Thumb,
  Title,
} from "./cardParts.tsx";
import { MapResult } from "./MapView.tsx";

// ------------------------------------------------------------- result cards

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
            className="mt-1.5 line-clamp-2 text-sm leading-relaxed text-ink-2"
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

export function VideoCard({ eager, result, globals }: CardProps) {
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
          {!result.thumbnail && hasMedia && previewOpen ? (
            <div className="mt-2 animate-fade-in">
              <MediaPreview src={result.iframe_src ?? ""} video />
            </div>
          ) : null}
          <p
            className="mt-1.5 line-clamp-2 text-sm leading-relaxed text-ink-2"
            dangerouslySetInnerHTML={{ __html: result.content_html || t("no_description") }}
            dir="auto"
          />
        </div>
        {result.thumbnail ? (
          // the player replaces the thumbnail in place - same behaviour as
          // the video grid, never expanding below the text
          <div className={`relative shrink-0 transition-all ${previewOpen && hasMedia ? "w-72" : "w-40"}`}>
            <div className="relative aspect-video overflow-hidden rounded-xl bg-surface-2">
              <ResultLink className="block size-full" globals={globals} result={result}>
                <Thumb
                  alt={result.title_text}
                  className="size-full"
                  eager={eager}
                  lengthDisplay={formatLength(result.length_display, result.length_seconds)}
                  src={result.thumbnail}
                />
              </ResultLink>
              {hasMedia && previewOpen ? (
                <>
                  <iframe
                    allowFullScreen
                    className="absolute inset-0 size-full"
                    referrerPolicy="origin"
                    src={result.iframe_src ?? ""}
                    title={result.title_text}
                  />
                  <button
                    aria-label={t("hide_video")}
                    className="absolute end-1 top-1 z-10 grid size-7 place-items-center rounded-full bg-black/70 text-white transition-colors hover:bg-accent-strong hover:text-ink"
                    onClick={() => {
                      setPreviewOpen(false);
                    }}
                    title={t("hide_video")}
                    type="button"
                  >
                    <CloseIcon className="size-3.5" />
                  </button>
                </>
              ) : hasMedia ? (
                <button
                  aria-label={t("play")}
                  className="absolute inset-0 z-10 grid size-full place-items-center rounded-xl bg-black/0 text-white transition-colors hover:bg-black/30"
                  onClick={() => {
                    setPreviewOpen(true);
                  }}
                  title={t("play")}
                  type="button"
                >
                  <span className="grid size-9 place-items-center rounded-full bg-black/70 shadow-pop">
                    <PlayIcon className="size-4 translate-x-px" />
                  </span>
                </button>
              ) : null}
            </div>
          </div>
        ) : null}
      </div>
      <EnginesLine result={result} />
    </ResultArticle>
  );
}

/** News-intent layout: same link style as every card, compact snippet, 16:9 thumb. */
/** News layout: editorial list with the photo on the left (as opposed to the
    web-result thumb on the right) and the recency date leading the meta row. */
export function NewsCard({ result, globals }: CardProps) {
  return (
    <ResultArticle priority={result.priority}>
      <div className="flex gap-4">
        {result.thumbnail ? (
          <ResultLink className="shrink-0" globals={globals} result={result}>
            <Thumb alt={result.title_text} className="aspect-video w-40" src={result.thumbnail} />
          </ResultLink>
        ) : null}
        <div className="min-w-0 flex-1">
          <PrettyUrl globals={globals} result={result} />
          <div className="mt-1">
            <Title globals={globals} result={result} />
          </div>
          <div className="mt-1">
            <MetaLine result={result} />
          </div>
          {result.content_html ? (
            <p
              className="mt-1 line-clamp-2 text-sm leading-relaxed text-ink-2"
              dangerouslySetInnerHTML={{ __html: result.content_html }}
              dir="auto"
            />
          ) : null}
        </div>
      </div>
      <EnginesLine result={result} />
    </ResultArticle>
  );
}

/** File-share layout: the magnet link doubles as the card's primary action
    tile, health stats read as a colored seeder/leecher strip - the
    "transfer" reading of the files category. */
export function TorrentCard({ result, globals }: CardProps) {
  const t = useT();
  return (
    <ResultArticle priority={result.priority}>
      <div className="flex gap-4">
        {result.magnetlink ? (
          <a
            aria-label={t("magnet_link")}
            className="grid size-14 shrink-0 self-start place-items-center rounded-xl bg-accent-soft text-accent transition-colors hover:bg-accent-strong hover:text-accent-contrast"
            href={result.magnetlink}
            title={t("magnet_link")}
          >
            <MagnetIcon className="size-6" />
          </a>
        ) : (
          <span className="grid size-14 shrink-0 self-start place-items-center rounded-xl bg-surface-2 text-ink-3">
            <FileIcon className="size-6" />
          </span>
        )}
        <div className="min-w-0 flex-1">
          <PrettyUrl globals={globals} result={result} />
          <div className="mt-1">
            <Title globals={globals} result={result} />
          </div>
          <div className="mt-1.5 flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-ink-3">
            {result.seed !== undefined ? (
              <span className="inline-flex items-center gap-1">
                <ArrowUpIcon className="size-3.5 text-ok" />
                <span className="font-semibold text-ok">{result.seed}</span>
                {t("seeder")}
              </span>
            ) : null}
            {result.leech !== undefined ? (
              <span className="inline-flex items-center gap-1">
                <ArrowDownIcon className="size-3.5 text-danger" />
                <span className="font-semibold text-danger">{result.leech}</span>
                {t("leecher")}
              </span>
            ) : null}
            {result.filesize ? (
              <span className="inline-flex items-center gap-1">
                <FileIcon className="size-3.5" />
                {result.filesize}
              </span>
            ) : null}
            {result.files ? (
              <span className="inline-flex items-center gap-1">
                <PackageIcon className="size-3.5" />
                {result.files} {t("files")}
              </span>
            ) : null}
            {result.published_date ? (
              <span className="inline-flex items-center gap-1">
                <CalendarIcon className="size-3.5" />
                {formatDate(result.published_date)}
              </span>
            ) : null}
          </div>
          {result.content_html ? (
            <p
              className="mt-1.5 line-clamp-2 text-sm leading-relaxed text-ink-2"
              dangerouslySetInnerHTML={{ __html: result.content_html }}
              dir="auto"
            />
          ) : null}
          {result.torrentfile ? (
            <div className="mt-2 flex flex-wrap items-center gap-2 text-xs">
              <a
                className="inline-flex items-center gap-1.5 rounded-full bg-surface-2 px-3 py-1 text-ink-2 transition-colors hover:text-ink"
                href={result.torrentfile}
              >
                <DownloadIcon className="size-3.5" />
                {t("torrent_file")}
              </a>
            </div>
          ) : null}
          <EnginesLine result={result} />
        </div>
      </div>
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
              className="mt-1.5 line-clamp-2 text-sm leading-relaxed text-ink-2"
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
        <p
          className="mt-1.5 line-clamp-2 text-sm leading-relaxed text-ink-2"
          dangerouslySetInnerHTML={{ __html: result.content_html }}
          dir="auto"
        />
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

/** General file-download layout, sharing the transfer-card language of the
    torrent card: type icon tile, compact stat strip, primary action. */
export function FileCard({ result, globals }: CardProps) {
  const t = useT();
  const isMedia = result.mtype === "audio" || result.mtype === "video";
  const tileClass = isMedia ? "bg-accent-soft text-accent" : "bg-surface-2 text-ink-3";
  const tileIcon =
    result.mtype === "audio" ? (
      <MusicIcon className="size-6" />
    ) : result.mtype === "video" ? (
      <FilmIcon className="size-6" />
    ) : (
      <FileIcon className="size-6" />
    );
  const stats: Array<[string, string | undefined]> = [
    [t("author"), result.author],
    [t("filename"), result.filename],
    [t("filesize"), result.size],
    [t("date"), result.time],
    [t("type"), result.mimetype],
  ];
  return (
    <ResultArticle priority={result.priority}>
      <div className="flex gap-4">
        <span className={`grid size-14 shrink-0 self-start place-items-center rounded-xl ${tileClass}`}>
          {tileIcon}
        </span>
        <div className="min-w-0 flex-1">
          <PrettyUrl globals={globals} result={result} />
          <div className="mt-1">
            <Title globals={globals} result={result} />
          </div>
          <div className="mt-1.5 flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-ink-3">
            {stats
              .filter(([, value]) => Boolean(value))
              .map(([label, value]) => (
                <span className="inline-flex min-w-0 items-center gap-1" key={label}>
                  {label}:
                  <span className="truncate text-ink-2" dir="auto">
                    {value}
                  </span>
                </span>
              ))}
          </div>
          {result.abstract_html ? (
            <p
              className="mt-1.5 line-clamp-2 text-sm leading-relaxed text-ink-2"
              dangerouslySetInnerHTML={{ __html: result.abstract_html }}
              dir="auto"
            />
          ) : null}
          {result.content_html ? (
            <p
              className="mt-1 line-clamp-2 text-sm leading-relaxed text-ink-2"
              dangerouslySetInnerHTML={{ __html: result.content_html }}
              dir="auto"
            />
          ) : null}
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
                <div className="mt-2 flex max-w-md items-center gap-3 rounded-xl border border-line bg-surface px-3 py-2">
                  <MusicIcon className="size-4 shrink-0 text-accent" />
                  <audio className="h-8 w-full" controls preload="none" src={result.embedded} />
                </div>
              )
            ) : (
              <a
                className="mt-2 inline-flex items-center gap-1.5 rounded-full bg-accent-soft px-3 py-1 text-xs font-medium text-accent transition-colors hover:bg-accent-strong hover:text-accent-contrast"
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
        </div>
      </div>
    </ResultArticle>
  );
}

/** Scholarly layout (science intent, arxiv/pubmed/...): authors · venue ·
    date meta line, clamped abstract, PDF/HTML actions and a compact DOI
    link.  Non-paper results on a science page (e.g. pdb figures) degrade
    gracefully - the card simply renders without the paper-specific bits. */
export function PaperCard({ result, globals }: CardProps) {
  const t = useT();
  const venueBits: string[] = [];
  if (result.journal) {
    venueBits.push(result.journal);
  }
  if (result.volume) {
    venueBits.push(`vol. ${result.volume}`);
  }
  if (result.number) {
    venueBits.push(`no. ${result.number}`);
  }
  if (result.pages) {
    venueBits.push(`pp. ${result.pages}`);
  }
  let authors = result.author ?? "";
  if (result.authors && result.authors.length > 0) {
    authors = result.authors.slice(0, 3).join(", ");
    if (result.authors.length > 3) {
      authors += ` ${t("et_al")}`;
    }
  }
  const metaBits: Array<string | null | ReactNode> = [
    result.published_date ? (
      <span className="inline-flex items-center gap-1" key="date">
        <CalendarIcon className="size-3" />
        {formatDate(result.published_date)}
      </span>
    ) : null,
    authors || null,
    venueBits.length > 0 ? venueBits.join(", ") : null,
  ];
  // every optional extra (PDF/HTML, DOI, citation note, subject tags) lives
  // in ONE footer row in a fixed order, so cards share the same skeleton no
  // matter which fields the source engine provides
  const maxTags = 2;
  const tags = result.tags ?? [];
  const [tagsExpanded, setTagsExpanded] = useState(false);
  const hasFooter = Boolean(result.pdf_url || result.html_url || result.doi || result.comments || tags.length > 0);
  return (
    <ResultArticle priority={result.priority}>
      <div className="flex gap-4">
        <div className="min-w-0 flex-1">
          <PrettyUrl globals={globals} result={result} />
          <div className="mt-1">
            <Title globals={globals} result={result} />
          </div>
          {metaBits.some(Boolean) ? (
            <p className="mt-1 truncate text-xs text-ink-3" dir="auto">
              {metaBits.filter(Boolean).map((bit, index) => (
                <span key={index}>
                  {index > 0 ? <span className="text-ink-3"> · </span> : null}
                  {bit}
                </span>
              ))}
            </p>
          ) : null}
          {result.content_html ? (
            <p
              className="mt-1.5 line-clamp-2 text-sm leading-relaxed text-ink-2"
              dangerouslySetInnerHTML={{ __html: result.content_html }}
              dir="auto"
            />
          ) : null}
          {hasFooter ? (
            // single line by default so every card shares the same height;
            // long comments/DOIs truncate instead of wrapping items onto
            // extra rows.  Wrapping returns only when tags are expanded.
            <div
              className={`mt-2 flex min-h-6 min-w-0 items-center gap-x-3 text-xs ${
                tagsExpanded ? "flex-wrap" : "overflow-hidden whitespace-nowrap"
              }`}
            >
              {result.pdf_url ? (
                <a
                  className="inline-flex shrink-0 items-center gap-1.5 rounded-full bg-accent-soft px-3 py-1 font-medium text-accent transition-colors hover:bg-accent-strong hover:text-accent-contrast"
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
                  className="inline-flex shrink-0 items-center gap-1.5 rounded-full bg-surface-2 px-3 py-1 text-ink-2 hover:text-ink"
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
                  className="max-w-36 shrink-0 truncate font-mono text-[11px] text-ink-3 hover:text-accent"
                  dir="ltr"
                  href={`https://${globals.doi_resolver}/${result.doi}`}
                  rel="noreferrer"
                  target="_blank"
                >
                  DOI {result.doi}
                </a>
              ) : null}
              {result.comments ? (
                <span className="min-w-0 truncate text-ink-3 italic" dir="auto" title={result.comments}>
                  {result.comments}
                </span>
              ) : null}
              {tags.slice(0, tagsExpanded ? tags.length : maxTags).map((tag) => (
                <span className="max-w-40 shrink-0 truncate text-ink-3" key={tag} title={tag}>
                  #{tag}
                </span>
              ))}
              {tags.length > maxTags && !tagsExpanded ? (
                <button
                  className="shrink-0 text-ink-3 transition-colors hover:text-ink"
                  onClick={() => {
                    setTagsExpanded(true);
                  }}
                  title={tags.join(", ")}
                  type="button"
                >
                  +{tags.length - maxTags}
                </button>
              ) : null}
              {tags.length > maxTags && tagsExpanded ? (
                <button
                  className="shrink-0 text-ink-3 transition-colors hover:text-ink"
                  onClick={() => {
                    setTagsExpanded(false);
                  }}
                  type="button"
                >
                  {t("show_less")}
                </button>
              ) : null}
            </div>
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

/** Dictionary entry card for the dictionaries/define categories: the headword
    is the identity (no URL chrome), a phonetic chip is lifted out of the
    Wiktionary blob ("IPA(key): /…/" is stable Wiktionary markup). */
export function DictionaryCard({ result, globals }: CardProps) {
  const ipa = /IPA\(key\):\s*\/([^/]+)\//.exec(result.content_html ?? "");
  const content = ipa ? (result.content_html ?? "").replaceAll(ipa[0], "") : result.content_html;
  return (
    <ResultArticle priority={result.priority}>
      <div className="mt-1 flex flex-wrap items-baseline gap-x-3 gap-y-1">
        <h3 className="text-lg font-semibold leading-snug">
          <ResultLink
            className="text-ink decoration-accent/50 underline-offset-2 hover:text-accent hover:underline"
            globals={globals}
            result={result}
          >
            <span dangerouslySetInnerHTML={{ __html: result.title_html }} dir="auto" />
          </ResultLink>
        </h3>
        {ipa ? (
          <code className="rounded bg-surface-2 px-1.5 py-0.5 font-mono text-xs text-ink-2" dir="ltr">
            /{ipa[1]}/
          </code>
        ) : null}
      </div>
      {content ? (
        <p
          className="mt-1.5 line-clamp-3 text-sm leading-relaxed text-ink-2"
          dangerouslySetInnerHTML={{ __html: content }}
          dir="auto"
        />
      ) : null}
      <EnginesLine result={result} />
    </ResultArticle>
  );
}

export function PackageCard({ result, globals }: CardProps) {
  const t = useT();
  const [tagsExpanded, setTagsExpanded] = useState(false);
  // same slot rhythm as DefaultCard: url / title / meta / content / engines;
  // secondary links fold into the engines row so no card grows extra rows
  const links: Array<{ icon: ReactNode; label: string; url: string }> = [];
  if (result.homepage) {
    links.push({ icon: <ExternalLinkIcon className="size-3" />, label: "Homepage", url: result.homepage });
  }
  if (result.source_code_url && result.source_code_url !== result.url) {
    links.push({ icon: <CodeIcon className="size-3" />, label: "Source code", url: result.source_code_url });
  }
  for (const [name, url] of Object.entries(result.project_links ?? {})) {
    if (url !== result.url) {
      links.push({ icon: <ExternalLinkIcon className="size-3" />, label: name, url });
    }
  }
  return (
    <ResultArticle priority={result.priority}>
      <PrettyUrl globals={globals} result={result} />
      <div className="mt-1">
        <Title globals={globals} result={result} />
      </div>
      {result.published_date || result.maintainer || result.popularity || result.license_name || result.version ? (
        <div className="mt-1 flex flex-wrap items-center gap-x-3 gap-y-0.5 text-xs text-ink-3">
          {result.published_date ? (
            <span className="inline-flex items-center gap-1" key="date">
              <CalendarIcon className="size-3" />
              {formatDate(result.published_date)}
            </span>
          ) : null}
          {result.maintainer ? (
            <span className="inline-flex min-w-0 items-center gap-1" key="author">
              {t("author")}:
              <span className="truncate text-ink-2" dir="auto">
                {result.maintainer}
              </span>
            </span>
          ) : null}
          {result.popularity ? (
            <span className="inline-flex items-center gap-1" key="popularity">
              <StarIcon className="size-3" />
              <span className="text-ink-2">{result.popularity}</span>
            </span>
          ) : null}
          {result.license_name ? (
            <span className="inline-flex items-center gap-1" key="license">
              {t("license")}:
              <span className="text-ink-2">
                {result.license_url ? (
                  <a
                    className="hover:text-accent hover:underline"
                    href={result.license_url}
                    rel="noreferrer"
                    target="_blank"
                  >
                    {result.license_name}
                  </a>
                ) : (
                  result.license_name
                )}
              </span>
            </span>
          ) : null}
          {result.version ? <span key="version">v{result.version}</span> : null}
        </div>
      ) : null}
      {result.content_html ? (
        <p
          className="mt-1.5 line-clamp-2 text-sm leading-relaxed text-ink-2"
          dangerouslySetInnerHTML={{ __html: result.content_html }}
          dir="auto"
        />
      ) : null}
      {result.tags && result.tags.length > 0 ? (
        <div className="mt-1.5 flex flex-wrap items-center gap-x-2.5 gap-y-1 text-xs text-ink-3">
          {result.tags.slice(0, 4).map((tag) => (
            <span className="max-w-48 truncate" key={tag} title={tag}>
              #{tag}
            </span>
          ))}
          {result.tags.length > 4 && !tagsExpanded ? (
            <button
              className="transition-colors hover:text-ink"
              onClick={() => {
                setTagsExpanded(true);
              }}
              title={result.tags.join(", ")}
              type="button"
            >
              +{result.tags.length - 4}
            </button>
          ) : null}
          {result.tags.length > 4 && tagsExpanded ? (
            <button
              className="transition-colors hover:text-ink"
              onClick={() => {
                setTagsExpanded(false);
              }}
              type="button"
            >
              {t("show_less")}
            </button>
          ) : null}
        </div>
      ) : null}
      <EnginesLine
        leading={
          links.length > 0
            ? links.map((link) => (
                <a
                  className="inline-flex items-center gap-1 rounded-full bg-surface-2 px-2 py-0.5 text-ink-2 transition-colors hover:text-ink"
                  href={link.url}
                  key={link.url}
                  rel="noreferrer"
                  target="_blank"
                >
                  {link.icon}
                  {link.label}
                </a>
              ))
            : null
        }
        result={result}
      />
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
        <p
          className="mt-1.5 line-clamp-2 text-sm leading-relaxed text-ink-2"
          dangerouslySetInnerHTML={{ __html: result.content_html }}
          dir="auto"
        />
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

/** Kagi-style video tiles for video-only result pages.  Tiles with an
    embeddable source get a Spotify-style hover play button that expands the
    player in place.  Cells carry data-hotkey-index so the results hotkeys
    can walk the grid like the list layouts. */
export const ResultCard = memo(function ResultCard(props: CardProps) {
  const { result } = props;
  switch (result.template) {
    case "images":
      return <DefaultCard {...props} />;
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
});
// memo: list items only change selection styling on their wrapper (the page
// renders the ring there), so unchanged props let hotkey navigation skip the
// whole card subtree
