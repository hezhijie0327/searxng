// SPDX-License-Identifier: Apache-2.0 WITH Commons-Clause-1.0

import { LoaderCircle } from "lucide-react";
import { Suspense } from "react";
import { BrandMark } from "./components/Brand.tsx";
import { Shell } from "./components/Shell.tsx";
import { I18nContext, useT } from "./lib/i18n.ts";
import { OverlayProvider } from "./lib/overlay.tsx";
import { RouterProvider, useRouter } from "./lib/router.tsx";
import type { ClientSettings } from "./lib/settings.ts";
import { SettingsContext } from "./lib/settings.ts";
import type { AnyPageData } from "./lib/types.ts";
import { isInfoPageData, isPreferencesPageData, isSearchPageData, isStatsPageData } from "./lib/types.ts";
import { IndexPage } from "./pages/IndexPage.tsx";
import { InfoPage, PreferencesPage, StatsPage } from "./pages/lazyPages.ts";
import { ResultsPage } from "./pages/ResultsPage.tsx";

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
    return (
      <Suspense fallback={<PageFallback />}>
        <PreferencesPage data={data} />
      </Suspense>
    );
  }
  if (isStatsPageData(data)) {
    return (
      <Suspense fallback={<PageFallback />}>
        <StatsPage data={data} />
      </Suspense>
    );
  }
  if (isInfoPageData(data)) {
    return (
      <Suspense fallback={<PageFallback />}>
        <InfoPage data={data} />
      </Suspense>
    );
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

function PageFallback() {
  return (
    <div className="grid min-h-[60vh] place-items-center">
      <LoaderCircle className="size-6 animate-spin-slow text-ink-3" />
    </div>
  );
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
