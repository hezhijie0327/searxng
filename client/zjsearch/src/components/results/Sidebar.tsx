// SPDX-License-Identifier: AGPL-3.0-or-later

import { type ReactNode, useState } from "react";
import { useT } from "../../lib/i18n.ts";
import { useOverlay } from "../../lib/overlay.tsx";
import type { GlobalData, InfoboxData, SearchPageData } from "../../lib/types.ts";
import { ChevronDownIcon, DownloadIcon, ExternalLinkIcon } from "../icons.tsx";
import { Link } from "../Shell.tsx";

function Box({ title, children, open = false }: { title: string; children: ReactNode; open?: boolean }) {
  return (
    <section className="overflow-hidden rounded-2xl border border-line bg-surface">
      <details open={open}>
        <summary className="cursor-pointer select-none px-4 py-2.5 text-xs font-semibold tracking-wide text-ink-3 uppercase transition-colors hover:text-ink">
          {title}
        </summary>
        <div className="px-4 pb-3">{children}</div>
      </details>
    </section>
  );
}

export function Infobox({
  infobox,
  globals,
  onSearch,
}: {
  infobox: InfoboxData;
  globals: GlobalData;
  onSearch: (q: string) => void;
}) {
  const t = useT();
  const [expanded, setExpanded] = useState(false);
  return (
    <div className="rounded-2xl border border-line bg-surface p-3">
      <div className={infobox.img_src ? "flex items-start gap-4" : ""}>
        {infobox.img_src ? (
          <img
            alt={infobox.title}
            className="aspect-square w-32 shrink-0 rounded-lg object-cover sm:w-36"
            decoding="async"
            loading="lazy"
            src={infobox.img_src}
          />
        ) : null}
        <h3 className="min-w-0 text-xl font-semibold leading-tight tracking-tight text-ink" dir="auto">
          {infobox.title}
        </h3>
      </div>

      <div className={`relative mt-3 ${expanded ? "" : "max-h-72 overflow-hidden"}`}>
        {infobox.attributes && infobox.attributes.length > 0 ? (
          <dl className="space-y-1 text-xs">
            {infobox.attributes.map((attribute, index) => (
              <div className="flex gap-2" key={index}>
                <dt className="shrink-0 text-ink-3">{attribute.label}:</dt>
                <dd className="min-w-0 text-ink-2">
                  {attribute.image_src ? (
                    <img
                      alt={attribute.image_alt}
                      className="inline-block max-h-24 rounded-lg align-middle"
                      decoding="async"
                      loading="lazy"
                      src={attribute.image_src}
                    />
                  ) : (
                    <span dir="auto">{attribute.value}</span>
                  )}
                </dd>
              </div>
            ))}
          </dl>
        ) : null}

        {infobox.content_html ? (
          <div
            className="mt-3 text-[13px] leading-relaxed text-ink-2 [&_a]:text-accent [&_a]:underline [&_a]:decoration-accent/40 [&_a]:underline-offset-2"
            dangerouslySetInnerHTML={{ __html: infobox.content_html }}
            dir="auto"
          />
        ) : null}

        {infobox.urls && infobox.urls.length > 0 ? (
          <ul className="mt-3 space-y-1 text-xs">
            {infobox.urls.map((url) => (
              <li className="truncate" key={url.url}>
                <a
                  className="inline-flex items-center gap-1 text-accent underline decoration-accent/40 underline-offset-2 hover:decoration-accent"
                  {...(globals.results_on_new_tab
                    ? { target: "_blank", rel: "noopener noreferrer" }
                    : { rel: "noreferrer" })}
                  href={url.url}
                >
                  <span className="truncate">{url.title}</span>
                  <ExternalLinkIcon className="size-3 shrink-0" />
                </a>
              </li>
            ))}
          </ul>
        ) : null}

        {infobox.related_topics && infobox.related_topics.length > 0 ? (
          <div className="mt-4 space-y-2">
            {infobox.related_topics.map((topic) => (
              <div key={topic.name}>
                <h4 className="text-xs font-medium text-ink" dir="auto">
                  {topic.name}
                </h4>
                <div className="mt-1 flex flex-wrap gap-1.5">
                  {topic.suggestions.map((suggestion) => (
                    <button
                      className="rounded-full bg-surface-2 px-2.5 py-1 text-xs text-ink-2 transition-colors hover:bg-accent-soft hover:text-accent"
                      key={suggestion}
                      onClick={() => {
                        onSearch(suggestion);
                      }}
                      type="button"
                    >
                      {suggestion}
                    </button>
                  ))}
                </div>
              </div>
            ))}
          </div>
        ) : null}
        {expanded ? null : (
          <div className="pointer-events-none absolute inset-x-0 bottom-0 h-10 bg-gradient-to-t from-surface to-transparent" />
        )}
      </div>
      <button
        className="mt-2 flex w-full items-center justify-center gap-1 border-t border-line pt-2.5 text-xs text-ink-3 transition-colors hover:text-ink"
        onClick={() => {
          setExpanded((value) => !value);
        }}
        type="button"
      >
        {expanded ? t("collapse") : t("expand")}
        <ChevronDownIcon className={`size-3.5 transition-transform ${expanded ? "rotate-180" : ""}`} />
      </button>
    </div>
  );
}

export function DebugPanels({ data }: { data: SearchPageData }) {
  const t = useT();
  const { openOverlay } = useOverlay();
  const hasEngineMsg = data.unresponsive_engines.length > 0 || data.timings.length > 0;
  const maxTime = data.max_response_time ?? 0;
  const globals = data.globals;
  return (
    <>
      {hasEngineMsg ? (
        <Box
          open={data.results.length === 0}
          title={
            data.max_response_time
              ? `${t("response_time")}: ${Math.round(data.max_response_time * 10) / 10} ${t("seconds")}`
              : t("engines_messages")
          }
        >
          {data.unresponsive_engines.length > 0 ? (
            <table className="w-full text-xs">
              <tbody>
                {data.unresponsive_engines.map(([name, errorMessage]) => (
                  <tr key={name}>
                    <td className="py-0.5 pr-2 align-top">
                      <button
                        className="text-left font-medium text-ink-2 hover:text-accent"
                        onClick={() => {
                          openOverlay(`/stats?engine=${encodeURIComponent(name)}`, t("engine_stats"));
                        }}
                        type="button"
                      >
                        {name}
                      </button>
                    </td>
                    <td className="py-0.5 text-danger" dir="auto">
                      {errorMessage}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : null}
          {data.timings.length > 0 ? (
            <table className="mt-1 w-full text-xs">
              <tbody>
                {data.timings.map((timing) => (
                  <tr key={timing.name}>
                    <td className="w-24 py-0.5 pr-2 truncate">
                      <button
                        className="text-left text-ink-2 hover:text-accent"
                        onClick={() => {
                          openOverlay(`/stats?engine=${encodeURIComponent(timing.name)}`, t("engine_stats"));
                        }}
                        type="button"
                      >
                        {timing.name}
                      </button>
                    </td>
                    <td className="py-0.5">
                      <div className="flex items-center gap-2">
                        <span className="w-10 shrink-0 text-right text-ink-3">{Math.round(timing.time * 10) / 10}</span>
                        <span className="h-1.5 flex-1 overflow-hidden rounded-full bg-surface-2">
                          <span
                            className="block h-full rounded-full bg-accent/70"
                            style={{ width: maxTime > 0 ? `${Math.max(2, (timing.time / maxTime) * 100)}%` : "0%" }}
                          />
                        </span>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : null}
        </Box>
      ) : null}

      {globals.search_formats.length > 0 ? (
        <Box title={t("download_results")}>
          <div className="flex flex-wrap gap-1.5">
            {globals.search_formats.map((format) => (
              <a
                className="inline-flex items-center gap-1 rounded-full bg-surface-2 px-3 py-1 text-xs text-ink-2 transition-colors hover:bg-accent-soft hover:text-accent"
                href={`${window.location.pathname}${window.location.search}${window.location.search.includes("?") ? "&" : "?"}format=${format}`}
                key={format}
              >
                <DownloadIcon className="size-3.5" />
                {format}
              </a>
            ))}
          </div>
        </Box>
      ) : null}
    </>
  );
}

export function Sidebar({ data, onSearch }: { data: SearchPageData; onSearch: (q: string) => void }) {
  const t = useT();
  const globals = data.globals;
  const hasInfobox = data.infoboxes.length > 0;
  const searchUrl = window.location.href;

  return (
    <aside className="flex flex-col gap-3">
      {hasInfobox ? (
        <section aria-label={t("info")} className="flex flex-col gap-3">
          {data.infoboxes.map((infobox, index) => (
            <Infobox globals={globals} infobox={infobox} key={index} onSearch={onSearch} />
          ))}
        </section>
      ) : null}

      <DebugPanels data={data} />

      {globals.method === "POST" ? (
        <Box title={t("search_url")}>
          <div className="flex items-start gap-2">
            <pre
              className="min-w-0 flex-1 overflow-x-auto rounded-lg bg-surface-2 p-2 font-mono text-xs leading-relaxed break-all whitespace-pre-wrap text-ink-2"
              dir="ltr"
            >
              {searchUrl}
            </pre>
            <CopyButton text={searchUrl} />
          </div>
        </Box>
      ) : null}
    </aside>
  );
}

export function CopyButton({ text, label }: { text: string; label?: string }) {
  const t = useT();
  const [copied, setCopied] = useState(false);
  return (
    <button
      className="shrink-0 rounded-lg bg-surface-2 px-2.5 py-1.5 text-xs text-ink-2 transition-colors hover:text-ink"
      onClick={() => {
        void navigator.clipboard
          .writeText(text)
          .then(() => {
            setCopied(true);
            window.setTimeout(() => {
              setCopied(false);
            }, 1500);
          })
          .catch(() => {
            /* clipboard unavailable */
          });
      }}
      type="button"
    >
      {copied ? t("copied") : (label ?? t("copy"))}
    </button>
  );
}
