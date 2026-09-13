// SPDX-License-Identifier: Apache-2.0 WITH Commons-Clause-1.0

import { AlertTriangle } from "lucide-react";
import type { ReactNode } from "react";
import { loadEngineDescriptions } from "../../lib/engineDescriptions.ts";
import { useT } from "../../lib/i18n.ts";
import type { PreferencesPageData } from "../../lib/types.ts";
import { EngineTooltip, reliabilityColor, Switch } from "./parts.tsx";

export function EnginesTab({
  tab,
  enabled,
  toggleEngine,
  showMetrics,
}: {
  tab: PreferencesPageData["engine_tabs"][number];
  enabled: Record<string, boolean>;
  toggleEngine: (key: string, value: boolean) => void;
  showMetrics: boolean;
}) {
  const t = useT();
  return (
    <div className="overflow-x-auto rounded-2xl border border-line">
      <table className="w-full min-w-[680px] text-left text-xs">
        <thead className="bg-surface-2 text-ink-3">
          <tr>
            <th className="px-4 py-3 font-medium">{t("allow")}</th>
            <th className="px-4 py-3 font-medium">{t("engine_name")}</th>
            <th className="px-4 py-3 font-medium">{t("bang")}</th>
            <th className="px-4 py-3 font-medium">{t("safesearch")}</th>
            <th className="px-4 py-3 font-medium">{t("time_range")}</th>
            <th className="px-4 py-3 font-medium">{t("weight")}</th>
            {showMetrics ? <th className="px-4 py-3 font-medium">{t("response_time")}</th> : null}
            <th className="px-4 py-3 font-medium">{t("max_time")}</th>
            {showMetrics ? <th className="px-4 py-3 font-medium">{t("reliability")}</th> : null}
          </tr>
        </thead>
        <tbody>
          {tab.groups.flatMap((group) => {
            const rows: ReactNode[] = [];
            if (group.engines.length > 1) {
              rows.push(
                <tr className="bg-surface-2/60" key={`group-${group.group}`}>
                  <td className="px-3 py-1.5 font-medium text-ink-2" colSpan={2}>
                    {group.group}
                  </td>
                  <td className="px-3 py-1.5" colSpan={showMetrics ? 7 : 5}>
                    {group.group_bang ? <code className="rounded bg-surface-2 px-1">{group.group_bang}</code> : null}
                  </td>
                </tr>,
              );
            }
            for (const engine of group.engines) {
              const key = `${engine.name}__${tab.category}`;
              rows.push(
                <tr className="border-t border-line transition-colors hover:bg-surface-2/40" key={key}>
                  <td className="px-3 py-3">
                    <Switch
                      checked={enabled[key] ?? false}
                      label={`${t("allow")} ${engine.name}`}
                      onChange={(value) => {
                        toggleEngine(key, value);
                      }}
                    />
                  </td>
                  <td className="max-w-52 px-4 py-3">
                    {/* hover OR keyboard focus reveals the tooltip */}
                    <div className="group/engine relative">
                      <button
                        aria-label={`${t("show_engine_info")}: ${engine.name}`}
                        className="flex items-center gap-1 truncate font-medium text-ink"
                        onMouseEnter={() => void loadEngineDescriptions()}
                        type="button"
                      >
                        {engine.enable_http ? <AlertTriangle className="size-3.5 shrink-0 text-warning" /> : null}
                        <span className="truncate">
                          {engine.name}
                          {engine.language ? ` (${engine.language})` : ""}
                        </span>
                      </button>
                      <EngineTooltip engine={engine} />
                    </div>
                  </td>
                  <td className="px-4 py-3">
                    <code className="rounded bg-surface-2 px-1">!{engine.shortcut}</code>
                  </td>
                  <td className="px-4 py-3">
                    {engine.supports_safesearch ? "✓" : <span className="text-ink-3">–</span>}
                  </td>
                  <td className="px-4 py-3">
                    {engine.supports_time_range ? "✓" : <span className="text-ink-3">–</span>}
                  </td>
                  <td className="px-4 py-3">{engine.weight}</td>
                  {showMetrics ? (
                    <td className="px-4 py-3">
                      {engine.stats_time !== null ? (
                        <div className="flex items-center gap-2">
                          <span className="w-10 text-ink-2">{engine.stats_time}</span>
                          <span className="h-1.5 w-24 overflow-hidden rounded-full bg-surface-2">
                            <span
                              className="block h-full bg-accent-strong"
                              style={{ width: `${Math.min(100, engine.stats_time)}%` }}
                            />
                          </span>
                        </div>
                      ) : (
                        <span className="text-ink-3">–</span>
                      )}
                    </td>
                  ) : null}
                  <td className={`px-4 py-3 ${engine.warn_timeout ? "font-medium text-danger" : "text-ink-2"}`}>
                    {engine.timeout}s
                  </td>
                  {showMetrics ? (
                    <td className={`px-4 py-3 font-medium ${reliabilityColor(engine.reliability)}`}>
                      {engine.reliability ?? "–"}
                    </td>
                  ) : null}
                </tr>,
              );
            }
            return rows;
          })}
        </tbody>
      </table>
    </div>
  );
}

// --------------------------------------------------------------------- page
