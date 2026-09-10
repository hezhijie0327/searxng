// SPDX-License-Identifier: AGPL-3.0-or-later

import { useState } from "react";
import { SearchBox } from "../components/SearchBox.tsx";
import { CategoryTabs, defaultFilterValues } from "../components/SearchControls.tsx";
import { Shell } from "../components/Shell.tsx";
import { useRouter } from "../lib/router.tsx";
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
    </Shell>
  );
}
