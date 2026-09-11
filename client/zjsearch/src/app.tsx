// SPDX-License-Identifier: AGPL-3.0-or-later

import { BrandMark } from "./components/icons.tsx";
import { Shell } from "./components/Shell.tsx";
import { I18nContext, useT } from "./lib/i18n.ts";
import { OverlayProvider } from "./lib/overlay.tsx";
import { RouterProvider, useRouter } from "./lib/router.tsx";
import type { ClientSettings } from "./lib/settings.ts";
import { SettingsContext } from "./lib/settings.ts";
import type { AnyPageData } from "./lib/types.ts";
import { isInfoPageData, isPreferencesPageData, isSearchPageData, isStatsPageData } from "./lib/types.ts";
import { IndexPage } from "./pages/IndexPage.tsx";
import { InfoPage } from "./pages/InfoPage.tsx";
import { PreferencesPage } from "./pages/PreferencesPage.tsx";
import { ResultsPage } from "./pages/ResultsPage.tsx";
import { StatsPage } from "./pages/StatsPage.tsx";

function Pages() {
  const { data, error } = useRouter();
  const t = useT();

  if (!data) {
    return (
      <div className="grid min-h-dvh place-items-center">
        <p className="text-sm text-ink-2">{error ?? "…"}</p>
      </div>
    );
  }

  const globals = data.globals;

  if (isSearchPageData(data)) {
    return <ResultsPage data={data} />;
  }

  if (isPreferencesPageData(data)) {
    return <PreferencesPage data={data} />;
  }
  if (isStatsPageData(data)) {
    return <StatsPage data={data} />;
  }
  if (isInfoPageData(data)) {
    return <InfoPage data={data} />;
  }
  switch (globals.page) {
    case "index":
      return <IndexPage data={data} />;
    default:
      return (
        <Shell globals={globals}>
          <main className="mx-auto flex w-full max-w-xl flex-1 flex-col items-center justify-center gap-4 px-4 pb-24 text-center animate-fade-up">
            <BrandMark className="size-14 rounded-[22%]" />
            <h1 className="text-5xl font-bold tracking-tight text-ink">404</h1>
            <p className="text-sm text-ink-2">{t("page_not_found")}</p>
          </main>
        </Shell>
      );
  }
}

export function App({ initialData, settings }: { initialData: AnyPageData | null; settings: ClientSettings }) {
  const locale = initialData?.globals.locale ?? "en";
  return (
    <SettingsContext.Provider value={settings}>
      <I18nContext.Provider value={locale}>
        <RouterProvider initialData={initialData}>
          <OverlayProvider>
            <Pages />
          </OverlayProvider>
        </RouterProvider>
      </I18nContext.Provider>
    </SettingsContext.Provider>
  );
}
