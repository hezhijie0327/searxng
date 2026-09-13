// SPDX-License-Identifier: Apache-2.0 WITH Commons-Clause-1.0

/** Secondary pages load on demand: the entry bundle stays lean and these
    chunks are only fetched the first time a drawer (or a direct visit)
    needs them. */

import { lazy } from "react";

export const InfoPage = lazy(() => import("./InfoPage.tsx").then((m) => ({ default: m.InfoPage })));

export const PreferencesPage = lazy(() =>
  import("./PreferencesPage.tsx").then((m) => ({ default: m.PreferencesPage })),
);

export const StatsPage = lazy(() => import("./StatsPage.tsx").then((m) => ({ default: m.StatsPage })));
