// SPDX-License-Identifier: Apache-2.0 WITH Commons-Clause-1.0

/**
 * The results meta line: 「found N results」 and 「took X s」 toggles, each
 * expanding its own strip in place — the result strip carries the utility
 * actions (copy the shareable URL, machine-readable download formats), the
 * engine strip the per-engine timings and errors.
 */

import {
  Check,
  ChevronDown,
  FileCode2,
  FileJson,
  FileSpreadsheet,
  Link2,
  List,
  type LucideIcon,
  Rss,
  Timer,
} from "lucide-react";
import { useState } from "react";
import { useOverlay } from "@/features/overlay/OverlayProvider.tsx";
import { useCopyFeedback } from "@/lib/clipboard.ts";
import { useT } from "@/lib/i18n.ts";
import type { SearchPageData } from "@/lib/types.ts";

/** per-format icons for the download strip; unknown configured formats fall
    back to the code-file icon */
const FORMAT_ICONS: Record<string, LucideIcon> = {
  csv: FileSpreadsheet,
  json: FileJson,
  rss: Rss,
  xml: FileCode2,
};

/** Display priority for the export chips (config order is arbitrary);
    unknown formats keep their config order at the end. */
const FORMAT_ORDER = ["rss", "json", "csv", "xml"];

function formatOrder(format: string): number {
  const index = FORMAT_ORDER.indexOf(format);
  return index === -1 ? FORMAT_ORDER.length : index;
}

const metaToggle = "inline-flex items-center gap-1 transition-colors hover:text-ink";
const stripChip =
  "inline-flex items-center gap-1 rounded-full bg-surface-2 px-3 py-1 text-xs text-ink-2 transition-colors hover:text-ink";

export function DebugPanels({
  data,
  resultCount,
  searchUrl,
}: {
  data: SearchPageData;
  /** number of rendered results (payload + appended infinite-scroll pages) */
  resultCount: number;
  /** shareable search URL: enables the copy action and the download formats
      (GET /search?format=…, machine-readable results) */
  searchUrl?: string;
}) {
  const t = useT();
  const { openOverlay } = useOverlay();
  const hasEnginesPanel = data.unresponsive_engines.length > 0 || data.timings.length > 0;
  // with zero results the engine messages matter most — start expanded
  const [openPanel, setOpenPanel] = useState<null | "engines" | "results">(() =>
    hasEnginesPanel && data.results.length === 0 ? "engines" : null,
  );
  const { copy, isCopied } = useCopyFeedback();
  const copied = isCopied(searchUrl ?? "");
  const toggle = (panel: "engines" | "results") => {
    setOpenPanel((current) => (current === panel ? null : panel));
  };
  const roundedTime = data.max_response_time ? Math.round(data.max_response_time * 10) / 10 : null;
  const maxTime = data.max_response_time ?? 0;

  return (
    <div>
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-ink-3">
        <button
          aria-expanded={openPanel === "results"}
          className={metaToggle}
          onClick={() => {
            toggle("results");
          }}
          type="button"
        >
          <List className="size-3 shrink-0" />
          {t("meta_found")} {resultCount} {t("meta_results")}
          <ChevronDown className={`size-3 transition-transform ${openPanel === "results" ? "rotate-180" : ""}`} />
        </button>
        {hasEnginesPanel ? (
          <button
            aria-expanded={openPanel === "engines"}
            className={metaToggle}
            onClick={() => {
              toggle("engines");
            }}
            type="button"
          >
            <Timer className="size-3 shrink-0" />
            {roundedTime !== null ? `${t("took")} ${roundedTime} ${t("seconds")}` : t("engines_messages")}
            <ChevronDown className={`size-3 transition-transform ${openPanel === "engines" ? "rotate-180" : ""}`} />
          </button>
        ) : null}
      </div>

      {openPanel === "results" ? (
        <div className="mt-2 rounded-2xl border border-line bg-surface px-4 py-3">
          <div className="flex items-center gap-3">
            <span
              className={`grid size-9 shrink-0 place-items-center rounded-full transition-colors ${
                copied ? "bg-ok/10 text-ok" : "bg-accent-soft text-accent"
              }`}
            >
              {copied ? <Check className="size-[18px]" /> : <Link2 className="size-[18px]" />}
            </span>
            <div className="min-w-0 flex-1">
              <p className="text-[13px] font-medium text-ink">{copied ? t("copied") : t("copy_search_url")}</p>
              {searchUrl ? (
                <p className="mt-0.5 truncate font-mono text-[11px] text-ink-3" dir="ltr" title={searchUrl}>
                  {searchUrl}
                </p>
              ) : null}
            </div>
            <button
              className={`shrink-0 rounded-full border px-3.5 py-1.5 text-xs font-medium transition-colors ${
                copied ? "border-ok/40 text-ok" : "border-line text-ink-2 hover:border-accent hover:text-accent"
              }`}
              onClick={() => {
                if (searchUrl) {
                  copy(searchUrl);
                }
              }}
              type="button"
            >
              {copied ? t("copied") : t("copy_link")}
            </button>
          </div>
          {searchUrl && data.globals.search_formats.length > 0 ? (
            <div className="mt-3 border-t border-line pt-3">
              <p className="text-[11px] font-medium text-ink-3">{t("export_formats")}</p>
              <div className="mt-2 flex flex-wrap gap-1.5">
                {[...data.globals.search_formats]
                  .sort((a, b) => formatOrder(a) - formatOrder(b))
                  .map((format) => {
                    const Icon = FORMAT_ICONS[format] ?? FileCode2;
                    return (
                      <a
                        className={stripChip}
                        href={`${searchUrl}&format=${format}`}
                        key={format}
                        rel="noreferrer"
                        target="_blank"
                      >
                        <Icon className="size-3 shrink-0 text-ink-3" />
                        {format.toUpperCase()}
                      </a>
                    );
                  })}
              </div>
            </div>
          ) : null}
        </div>
      ) : null}

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
