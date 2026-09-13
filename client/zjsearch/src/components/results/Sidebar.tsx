// SPDX-License-Identifier: Apache-2.0 WITH Commons-Clause-1.0

import { ChevronDown, ChevronLeft, ChevronRight, ExternalLink, Search } from "lucide-react";
import { type ReactNode, useCallback, useEffect, useId, useLayoutEffect, useRef, useState } from "react";
import { useT } from "../../lib/i18n.ts";
import { newTabLinkProps } from "../../lib/link.ts";
import { scrollBehavior } from "../../lib/motion.ts";
import { useOverlay } from "../../lib/overlay.tsx";
import type { GlobalData, InfoboxData, SearchPageData } from "../../lib/types.ts";
import { CopyButton } from "../CopyButton.tsx";

function Box({ title, children }: { title: string; children: ReactNode }) {
  return (
    <section className="overflow-hidden rounded-2xl border border-line bg-surface">
      <details>
        <summary className="cursor-pointer select-none px-4 py-2.5 text-xs font-semibold tracking-wide text-ink-3 uppercase transition-colors hover:text-ink">
          {title}
        </summary>
        <div className="px-4 pb-3">{children}</div>
      </details>
    </section>
  );
}

/** Collapsed infobox preview height (the old max-h-72 clamp). */
const INFOBOX_PREVIEW_PX = 288;

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
  const contentId = useId();
  const contentRef = useRef<HTMLDivElement>(null);
  // natural content height, kept current by a ResizeObserver: max-height
  // cannot transition to `none`, so the expanded state must pin a real
  // pixel value for the browser to animate to
  const [contentPx, setContentPx] = useState<number | null>(null);

  useEffect(() => {
    const el = contentRef.current;
    if (!el) {
      return;
    }
    const ro = new ResizeObserver(() => {
      // +1 guards against sub-pixel rounding clipping the last text line
      setContentPx(Math.ceil(el.getBoundingClientRect().height) + 1);
    });
    ro.observe(el);
    return () => {
      ro.disconnect();
    };
  }, []);

  // content that fits the preview needs no clamp, gradient or toggle
  const needsClamp = contentPx === null || contentPx > INFOBOX_PREVIEW_PX + 24;

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

      {/* animated disclosure: overflow-hidden stays on in both states (without
          it the inner mt-3 collapses through the wrapper when expanded and
          the visible content jumps up 12px on toggle); flow-root keeps the
          inner's first-child margin inside the measured box */}
      <div
        className="relative mt-3 overflow-hidden transition-[max-height] duration-300 ease-out"
        id={contentId}
        style={{
          maxHeight: needsClamp ? (expanded ? (contentPx ?? INFOBOX_PREVIEW_PX) : INFOBOX_PREVIEW_PX) : undefined,
        }}
      >
        <div className="relative flow-root" ref={contentRef}>
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
                        onError={(event) => {
                          event.currentTarget.style.display = "none";
                        }}
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
                    {...newTabLinkProps(globals.results_on_new_tab)}
                    href={url.url}
                  >
                    <span className="truncate">{url.title}</span>
                    <ExternalLink className="size-3 shrink-0" />
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
        </div>
        {needsClamp ? (
          <div
            className={`pointer-events-none absolute inset-x-0 bottom-0 h-10 bg-gradient-to-t from-surface to-transparent transition-opacity duration-300 ${
              expanded ? "opacity-0" : "opacity-100"
            }`}
          />
        ) : null}
      </div>
      {needsClamp ? (
        <button
          aria-controls={contentId}
          aria-expanded={expanded}
          className="mt-2 flex w-full items-center justify-center gap-1 border-t border-line pt-2.5 text-xs text-ink-3 transition-colors hover:text-ink"
          onClick={() => {
            setExpanded((value) => !value);
          }}
          type="button"
        >
          {expanded ? t("collapse") : t("expand")}
          <ChevronDown className={`size-3.5 transition-transform ${expanded ? "rotate-180" : ""}`} />
        </button>
      ) : null}
    </div>
  );
}

export function SuggestionsBox({ data, onSearch }: { data: SearchPageData; onSearch: (q: string) => void }) {
  const t = useT();
  const stripRef = useRef<HTMLDivElement>(null);
  const [canLeft, setCanLeft] = useState(false);
  const [canRight, setCanRight] = useState(false);

  const measure = useCallback(() => {
    const strip = stripRef.current;
    if (!strip) {
      return;
    }
    const maxScroll = strip.scrollWidth - strip.clientWidth;
    setCanLeft(strip.scrollLeft > 2);
    setCanRight(strip.scrollLeft < maxScroll - 2);
  }, []);

  // biome-ignore lint/correctness/useExhaustiveDependencies: re-measure when a new query swaps the suggestion set
  useLayoutEffect(() => {
    measure();
  }, [data.suggestions, measure]);

  useEffect(() => {
    window.addEventListener("resize", measure);
    return () => window.removeEventListener("resize", measure);
  }, [measure]);

  if (data.suggestions.length === 0) {
    return null;
  }

  const page = (direction: -1 | 1) => {
    const strip = stripRef.current;
    if (!strip) {
      return;
    }
    strip.scrollBy({ left: direction * strip.clientWidth * 0.8, behavior: scrollBehavior() });
  };

  // buttons are persistent so flipping state never shifts the chips
  const arrowClass =
    "flex size-7 shrink-0 items-center justify-center rounded-full text-ink-3 transition hover:bg-surface-2 hover:text-ink disabled:pointer-events-none disabled:opacity-30";
  return (
    <div className="flex items-center gap-1">
      <button
        aria-label={t("previous_page")}
        className={arrowClass}
        disabled={!canLeft}
        onClick={() => {
          page(-1);
        }}
        type="button"
      >
        <ChevronLeft className="size-3.5" />
      </button>
      <div
        className="flex min-w-0 flex-1 gap-1.5 overflow-x-auto [scrollbar-width:none] [&::-webkit-scrollbar]:hidden"
        onScroll={measure}
        ref={stripRef}
      >
        {data.suggestions.map((suggestion) => (
          <button
            className="inline-flex shrink-0 items-center gap-1.5 rounded-full bg-surface-2 px-3 py-1.5 text-[13px] text-ink-2 transition-colors hover:bg-accent-soft hover:text-accent"
            dir="auto"
            key={suggestion.q}
            onClick={() => {
              onSearch(suggestion.q);
            }}
            type="button"
          >
            <Search className="size-3.5 shrink-0 text-ink-3" />
            <span className="max-w-40 truncate">{suggestion.title}</span>
          </button>
        ))}
      </div>
      <button
        aria-label={t("next_page")}
        className={arrowClass}
        disabled={!canRight}
        onClick={() => {
          page(1);
        }}
        type="button"
      >
        <ChevronRight className="size-3.5" />
      </button>
    </div>
  );
}

export function DebugPanels({ data, leading }: { data: SearchPageData; leading?: ReactNode }) {
  const t = useT();
  const { openOverlay } = useOverlay();
  const hasEnginesPanel = data.unresponsive_engines.length > 0 || data.timings.length > 0;
  // with zero results the engine messages matter most — start expanded
  const [openPanel, setOpenPanel] = useState<null | "engines">(() =>
    hasEnginesPanel && data.results.length === 0 ? "engines" : null,
  );
  const roundedTime = data.max_response_time ? Math.round(data.max_response_time * 10) / 10 : null;
  const maxTime = data.max_response_time ?? 0;
  return (
    <div>
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-ink-3">
        {leading}
        {hasEnginesPanel ? (
          <button
            aria-expanded={openPanel === "engines"}
            className="inline-flex items-center gap-1 transition-colors hover:text-ink"
            onClick={() => {
              setOpenPanel((current) => (current === "engines" ? null : "engines"));
            }}
            type="button"
          >
            {roundedTime !== null ? `${t("took")} ${roundedTime} ${t("seconds")}` : t("engines_messages")}
            <ChevronDown className={`size-3 transition-transform ${openPanel === "engines" ? "rotate-180" : ""}`} />
          </button>
        ) : null}
      </div>

      {openPanel === "engines" ? (
        <div className="mt-2 rounded-2xl border border-line bg-surface px-4 py-3">
          {/* one table for every engine: timings get seconds + bar, unresponsive
              engines get their error label + an empty track on the same grid */}
          <table className="w-full text-xs">
            <tbody>
              {data.unresponsive_engines.map(([name, errorMessage]) => (
                <tr key={name}>
                  <td className="w-24 py-0.5 pr-2 truncate">
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
                  <td className="py-0.5">
                    <div className="flex items-center gap-2">
                      {/* right-aligned into the seconds column: the error's
                          right edge lines up with the digits / bar start */}
                      <span className="w-24 shrink-0 truncate text-right text-danger" dir="auto" title={errorMessage}>
                        {errorMessage}
                      </span>
                      <span className="h-1.5 flex-1 rounded-full bg-surface-2" />
                    </div>
                  </td>
                </tr>
              ))}
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
                      <span className="w-24 shrink-0 text-right text-ink-3">{Math.round(timing.time * 10) / 10}</span>
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
        </div>
      ) : null}
    </div>
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

      {globals.method === "POST" ? (
        <Box title={t("search_url")}>
          <div className="flex items-start gap-2">
            <pre
              className="min-w-0 flex-1 overflow-x-auto rounded-lg bg-surface-2 p-2 font-mono text-xs leading-relaxed break-all whitespace-pre-wrap text-ink-2"
              dir="ltr"
            >
              {searchUrl}
            </pre>
            <CopyButton value={searchUrl} />
          </div>
        </Box>
      ) : null}
    </aside>
  );
}
