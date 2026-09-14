// SPDX-License-Identifier: Apache-2.0 WITH Commons-Clause-1.0

import { Check, LayoutGrid, X } from "lucide-react";
import { useT } from "@/lib/i18n.ts";
import type { PreferencesPageData } from "@/lib/types.ts";
import { EnginesTab } from "@/pages/preferences/EnginesTab.tsx";
import { CategoryTab } from "@/pages/preferences/parts.tsx";
import type { PreferencesForm } from "@/pages/preferences/usePreferencesForm.ts";

/** Engines tab pane: per-category engine tables plus the enable/disable-all
    shortcuts for the visible category. */
export function EnginesPane({
  data,
  form,
  engineTab,
  onEngineTab,
}: {
  data: PreferencesPageData;
  form: PreferencesForm;
  engineTab: number;
  onEngineTab: (index: number) => void;
}) {
  const t = useT();
  const currentEngineTab = data.engine_tabs[engineTab];
  const tabEngineKeys = currentEngineTab
    ? currentEngineTab.groups.flatMap((group) =>
        group.engines.map((engine) => `${engine.name}__${currentEngineTab.category}`),
      )
    : [];
  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <p className="flex items-center gap-2 text-sm text-ink-2">
          <LayoutGrid className="size-4 text-ink-3" />
          {t("currently_used_engines")}
        </p>
        {currentEngineTab ? (
          <div className="flex items-center gap-1.5">
            <button
              className="inline-flex items-center gap-1.5 rounded-full border border-line px-3 py-1.5 text-[13px] text-ink-2 transition-colors hover:border-ok hover:text-ok"
              onClick={() => {
                form.setAllEngines(tabEngineKeys, true);
              }}
              type="button"
            >
              <Check className="size-3.5" />
              {t("enable_all")}
            </button>
            <button
              className="inline-flex items-center gap-1.5 rounded-full border border-line px-3 py-1.5 text-[13px] text-ink-2 transition-colors hover:border-danger hover:text-danger"
              onClick={() => {
                form.setAllEngines(tabEngineKeys, false);
              }}
              type="button"
            >
              <X className="size-3.5" />
              {t("disable_all")}
            </button>
          </div>
        ) : null}
      </div>
      <div className="flex flex-wrap items-center gap-x-1 gap-y-0.5">
        {data.engine_tabs.map((tabInfo, index) => (
          <CategoryTab
            active={index === engineTab}
            category={tabInfo.category}
            key={tabInfo.category}
            label={tabInfo.label}
            onClick={() => {
              onEngineTab(index);
            }}
          />
        ))}
      </div>
      {currentEngineTab ? (
        <EnginesTab
          enabled={form.engines}
          showMetrics={data.globals.enable_metrics}
          tab={currentEngineTab}
          toggleEngine={form.toggleEngine}
        />
      ) : null}
    </div>
  );
}
