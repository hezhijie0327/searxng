// SPDX-License-Identifier: Apache-2.0 WITH Commons-Clause-1.0

import { AlertTriangle, Check, Cookie, LayoutGrid, Shield, SlidersHorizontal, Sun, Terminal } from "lucide-react";
import { useMemo, useState } from "react";
import { Link, Shell } from "@/components/Shell.tsx";
import { useOverlay } from "@/features/overlay/OverlayProvider.tsx";
import { useT } from "@/lib/i18n.ts";
import type { PreferencesPageData } from "@/lib/types.ts";
import { CookiesTab } from "@/pages/preferences/tabs/CookiesTab.tsx";
import { EnginesPane } from "@/pages/preferences/tabs/EnginesPane.tsx";
import { GeneralTab } from "@/pages/preferences/tabs/GeneralTab.tsx";
import { PrivacyTab } from "@/pages/preferences/tabs/PrivacyTab.tsx";
import { QueryTab } from "@/pages/preferences/tabs/QueryTab.tsx";
import { UiTab } from "@/pages/preferences/tabs/UiTab.tsx";
import { usePreferencesForm } from "@/pages/preferences/usePreferencesForm.ts";

type PrefsTab = "general" | "ui" | "privacy" | "engines" | "query" | "cookies";

export function PreferencesPage({ data, embedded = false }: { data: PreferencesPageData; embedded?: boolean }) {
  const t = useT();
  const { openDocument } = useOverlay();
  const globals = data.globals;
  const form = usePreferencesForm(data);
  const locked = useMemo(() => new Set(data.locked_preferences), [data.locked_preferences]);

  const [tab, setTab] = useState<PrefsTab>("general");
  const [engineTab, setEngineTab] = useState(0);

  const isPreview = new URLSearchParams(window.location.search).get("preferences_preview_only") === "true";

  const tabs = [
    { id: "general", label: t("general"), icon: <SlidersHorizontal className="size-3.5" /> },
    { id: "ui", label: t("user_interface"), icon: <Sun className="size-3.5" /> },
    { id: "privacy", label: t("privacy"), icon: <Shield className="size-3.5" /> },
    { id: "engines", label: t("engines"), icon: <LayoutGrid className="size-3.5" /> },
    { id: "query", label: t("special_queries"), icon: <Terminal className="size-3.5" /> },
    { id: "cookies", label: t("cookies"), icon: <Cookie className="size-3.5" /> },
  ] as const;

  return (
    <Shell embedded={embedded} globals={globals}>
      <main className="mx-auto w-full max-w-4xl flex-1 px-4 pb-20 sm:px-6">
        {!embedded ? (
          <div className="flex items-center justify-between py-6">
            <h1 className="text-2xl font-semibold tracking-tight text-ink">{t("preferences")}</h1>
            <div className="flex items-center gap-3">
              {form.savedAt > 0 ? (
                <span className="inline-flex items-center gap-1 text-xs text-ok animate-fade-in">
                  <Check className="size-3.5" />
                  {t("saved")}
                </span>
              ) : null}
            </div>
          </div>
        ) : form.savedAt > 0 ? (
          <div className="flex justify-end py-3">
            <span className="inline-flex items-center gap-1 text-xs text-ok animate-fade-in">
              <Check className="size-3.5" />
              {t("saved")}
            </span>
          </div>
        ) : null}

        {isPreview ? (
          <div className="mb-4 flex items-start gap-3 rounded-2xl border border-warning/40 bg-warning/10 p-4 text-sm text-ink-2 animate-fade-up">
            <AlertTriangle className="mt-0.5 size-4 shrink-0 text-warning" />
            <div>
              <p>{t("preview_banner")}</p>
              <ul className="mt-1.5 list-disc ps-5">
                <li>{t("press_save_to_copy")}</li>
                <li>
                  {t("view_browser_prefs")}{" "}
                  <Link className="text-accent hover:underline" href="/preferences">
                    /preferences
                  </Link>
                </li>
              </ul>
            </div>
          </div>
        ) : null}

        <div className="mb-6 flex flex-wrap gap-1.5 rounded-2xl border border-line bg-surface p-2" role="tablist">
          {tabs.map((item) => (
            <button
              aria-controls={`prefs-panel-${item.id}`}
              aria-selected={tab === item.id}
              className={`flex flex-1 items-center justify-center gap-2 whitespace-nowrap rounded-xl px-4 py-2 text-[13px] transition-colors ${
                tab === item.id
                  ? "bg-accent-strong font-medium text-accent-contrast"
                  : "text-ink-2 hover:bg-surface-2 hover:text-ink"
              }`}
              id={`prefs-tab-${item.id}`}
              key={item.id}
              onClick={() => {
                setTab(item.id);
              }}
              role="tab"
              type="button"
            >
              {item.icon}
              <span>{item.label}</span>
            </button>
          ))}
        </div>

        <div aria-labelledby={`prefs-tab-${tab}`} id={`prefs-panel-${tab}`} key={tab} role="tabpanel">
          {tab === "general" ? <GeneralTab data={data} form={form} locked={locked} /> : null}
          {tab === "ui" ? <UiTab data={data} form={form} locked={locked} /> : null}
          {tab === "privacy" ? <PrivacyTab data={data} form={form} locked={locked} /> : null}
          {tab === "engines" ? (
            <EnginesPane data={data} engineTab={engineTab} form={form} onEngineTab={setEngineTab} />
          ) : null}
          {tab === "query" ? <QueryTab data={data} form={form} /> : null}
          {tab === "cookies" ? <CookiesTab data={data} form={form} /> : null}
        </div>
        <div className="mt-6 space-y-1 text-center text-xs text-ink-3">
          <p className="leading-5">
            {t("powered_by")}{" "}
            <a
              className="transition-colors hover:text-accent hover:underline"
              href={globals.git_url}
              rel="noreferrer"
              target="_blank"
            >
              SearXNG
            </a>
            {globals.version ? <span className="ms-1 opacity-70">v{globals.version}</span> : null}
          </p>
          <p className="leading-5">
            {t("license")}:{" "}
            <a
              className="transition-colors hover:text-accent hover:underline"
              href="/static/themes/zjsearch/LICENSE.txt"
              onClick={(event) => {
                event.preventDefault();
                openDocument(t("license"), "/static/themes/zjsearch/LICENSE.txt");
              }}
            >
              Apache-2.0 with Commons Clause v1.0
            </a>
          </p>
        </div>
      </main>
    </Shell>
  );
}
