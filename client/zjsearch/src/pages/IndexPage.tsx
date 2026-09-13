// SPDX-License-Identifier: Apache-2.0 WITH Commons-Clause-1.0

import { Lightbulb, SlidersHorizontal } from "lucide-react";
import { useState } from "react";
import { HelpModal } from "../components/HelpModal.tsx";
import { SearchBox } from "../components/SearchBox.tsx";
import { CategoryTabs, defaultFilterValues, type FilterValues, SearchFilters } from "../components/SearchControls.tsx";
import { Shell } from "../components/Shell.tsx";
import { type HotkeyTarget, useHotkeys } from "../features/hotkeys.ts";
import { useT } from "../lib/i18n.ts";
import { useRouter } from "../lib/router.tsx";
import { useSettings } from "../lib/settings.ts";
import type { BasicPageData } from "../lib/types.ts";

interface IndexData extends BasicPageData {
  selected_categories?: string[];
}

export function IndexPage({ data }: { data: IndexData }) {
  const { search } = useRouter();
  const globals = data.globals;
  const [query, setQuery] = useState("");
  const [selected, setSelected] = useState<string[]>(
    data.selected_categories && data.selected_categories.length > 0
      ? data.selected_categories
      : [globals.default_category],
  );
  const [filters, setFilters] = useState<FilterValues>(() => defaultFilterValues(globals));
  const [optionsOpen, setOptionsOpen] = useState(false);

  const submitSearch = (q: string, categories = selected) => {
    const trimmed = q.trim();
    if (!trimmed) {
      return;
    }
    search({
      q: trimmed,
      categories,
      language: filters.language,
      time_range: filters.time_range,
      safesearch: filters.safesearch,
      pageno: 1,
    });
  };

  const [helpOpen, setHelpOpen] = useState(false);
  const [hintHidden, setHintHidden] = useState(() => window.localStorage.getItem("zjs-hint-hidden") === "1");
  const settings = useSettings();
  const t = useT();

  // "?" opens the shortcuts help on the home page too; the result-navigation
  // keys have nothing to act on here
  const hotkeyTarget: HotkeyTarget = {
    move: () => {},
    open: () => {},
    yank: () => null,
    page: () => {},
    focusSearch: () => {
      (document.querySelector('input[name="q"]') as HTMLInputElement | null)?.focus();
    },
  };
  useHotkeys(settings.hotkeys, hotkeyTarget, () => {
    setHelpOpen((open) => !open);
  });

  return (
    <Shell globals={globals} variant="hero">
      <main className="mx-auto flex w-full max-w-2xl flex-col items-center px-4 pb-24">
        <h1 className="animate-fade-up text-6xl font-extrabold tracking-tight text-ink sm:text-7xl">
          {globals.instance_name}
          <span className="text-accent-strong">.</span>
        </h1>
        {/* raised stacking level: fade-up leaves a residual transform (a
            stacking context) on every animated sibling, which would let the
            category tabs paint over the z-30 dropdown */}
        <div className="relative z-10 mt-12 w-full animate-fade-up [animation-delay:60ms]">
          <SearchBox
            initialQuery=""
            onQueryChange={setQuery}
            onSubmitQuery={(q) => {
              submitSearch(q);
            }}
            query={query}
            variant="hero"
          />
        </div>
        {/* single stable toggle: opens the tabs + filter rows in flow; the
            hero is top-anchored (30vh), so growth extends downward only and
            the brand/search box never move */}
        <div className="mt-3 flex w-full justify-end animate-fade-up [animation-delay:120ms]">
          <button
            aria-expanded={optionsOpen}
            className={`inline-flex items-center gap-1.5 rounded-full px-3 py-1.5 text-[13px] transition-colors ${
              optionsOpen ? "bg-surface-2 text-ink" : "text-ink-3 hover:bg-surface-2 hover:text-ink"
            }`}
            onClick={() => {
              setOptionsOpen((open) => !open);
            }}
            type="button"
          >
            <SlidersHorizontal className="size-3.5" />
            {t("search_options")}
          </button>
        </div>
        {optionsOpen ? (
          <>
            <div className="relative z-10 mt-3 w-full animate-fade-up [animation-delay:60ms]">
              <CategoryTabs
                globals={globals}
                onSearch={(categories) => {
                  setSelected(categories);
                  submitSearch(query, categories);
                }}
                onSelectionChange={setSelected}
                selected={selected}
                wrap
              />
            </div>
            <div className="relative z-10 mt-2 flex w-full flex-wrap items-center gap-1.5 ps-6 animate-fade-in">
              <SearchFilters
                globals={globals}
                onChange={(next) => {
                  setFilters((prev) => ({ ...prev, ...next }));
                }}
                values={filters}
              />
            </div>
          </>
        ) : null}
      </main>
      {hintHidden ? null : (
        <div className="mx-auto mb-10 w-full max-w-xl px-4">
          <div className="flex items-center gap-3 rounded-2xl border border-line bg-surface px-4 py-2.5 text-sm animate-fade-up">
            <Lightbulb className="size-4 shrink-0 text-accent" />
            <button
              className="min-w-0 flex-1 truncate text-left text-ink-2 transition-colors hover:text-ink"
              onClick={() => {
                setHelpOpen(true);
              }}
              type="button"
            >
              {t("hotkeys_hint")}
            </button>
            <button
              aria-label={t("close")}
              className="shrink-0 rounded-lg px-2 py-1 text-xs text-ink-3 transition-colors hover:bg-surface-2 hover:text-ink"
              onClick={() => {
                window.localStorage.setItem("zjs-hint-hidden", "1");
                setHintHidden(true);
              }}
              type="button"
            >
              {t("close")}
            </button>
          </div>
        </div>
      )}
      {helpOpen ? <HelpModal layout={settings.hotkeys} onClose={() => setHelpOpen(false)} /> : null}
    </Shell>
  );
}
