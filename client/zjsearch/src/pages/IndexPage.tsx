// SPDX-License-Identifier: AGPL-3.0-or-later

import { useState } from "react";
import { HelpModal } from "../components/HelpModal.tsx";
import { LightbulbIcon } from "../components/icons.tsx";
import { SearchBox } from "../components/SearchBox.tsx";
import { CategoryTabs, defaultFilterValues } from "../components/SearchControls.tsx";
import { Shell } from "../components/Shell.tsx";
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

  const filters = defaultFilterValues(globals);

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

  return (
    <Shell globals={globals} variant="hero">
      <main className="mx-auto flex w-full max-w-2xl flex-col items-center px-4 pb-24">
        <h1 className="animate-fade-up text-6xl font-extrabold tracking-tight text-ink sm:text-7xl">
          {globals.instance_name}
          <span className="text-accent-strong">.</span>
        </h1>
        <div className="mt-12 w-full animate-fade-up [animation-delay:60ms]">
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
        <div className="mt-3 max-w-full animate-fade-up [animation-delay:120ms]">
          <CategoryTabs
            globals={globals}
            onSearch={(categories) => {
              setSelected(categories);
              submitSearch(query, categories);
            }}
            selected={selected}
          />
        </div>
      </main>
      {hintHidden ? null : (
        <div className="mx-auto mb-10 w-full max-w-xl px-4">
          <div className="flex items-center gap-3 rounded-2xl border border-line bg-surface px-4 py-2.5 text-sm animate-fade-up">
            <LightbulbIcon className="size-4 shrink-0 text-accent" />
            <button
              className="min-w-0 flex-1 truncate text-left text-ink-2 transition-colors hover:text-ink"
              onClick={() => {
                setHelpOpen(true);
              }}
              type="button"
            >
              Press <kbd className="rounded border border-line bg-surface-2 px-1.5 font-mono text-xs">?</kbd> anytime
              for keyboard shortcuts
            </button>
            <button
              aria-label="Close"
              className="shrink-0 rounded-lg px-2 py-1 text-xs text-ink-3 transition-colors hover:bg-surface-2 hover:text-ink"
              onClick={() => {
                window.localStorage.setItem("zjs-hint-hidden", "1");
                setHintHidden(true);
              }}
              type="button"
            >
              Close
            </button>
          </div>
        </div>
      )}
      {helpOpen ? <HelpModal layout={settings.hotkeys} onClose={() => setHelpOpen(false)} /> : null}
    </Shell>
  );
}
