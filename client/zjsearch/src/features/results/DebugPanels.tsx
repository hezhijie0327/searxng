// SPDX-License-Identifier: Apache-2.0 WITH Commons-Clause-1.0

import { ChevronDown, Timer } from "lucide-react";
import { type ReactNode, useState } from "react";
import { useOverlay } from "@/features/overlay/OverlayProvider.tsx";
import { useT } from "@/lib/i18n.ts";
import type { SearchPageData } from "@/lib/types.ts";

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
            <Timer className="size-3 shrink-0" />
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
