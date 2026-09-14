// SPDX-License-Identifier: Apache-2.0 WITH Commons-Clause-1.0

/**
 * Results right rail: knowledge (infoboxes) and diagnostics-only extras.
 * In POST mode it also offers the shareable search URL, since the query
 * never reaches the address bar there.
 */

import type { ReactNode } from "react";
import { ClickToCopy } from "@/components/CopyButton.tsx";
import { Infobox } from "@/features/results/Infobox.tsx";
import { useT } from "@/lib/i18n.ts";
import { buildSearchUrl } from "@/lib/searchParams.ts";
import type { SearchPageData } from "@/lib/types.ts";

function Box({ title, children }: { title: string; children: ReactNode }) {
  return (
    <section className="overflow-hidden rounded-2xl border border-line bg-surface">
      <details>
        <summary className="cursor-pointer select-none px-4 py-2.5 text-xs font-semibold tracking-wide text-ink-3 uppercase transition-colors hover:text-ink">
          {title}
        </summary>
        <div className="px-4 pb-3">{children}</div>
      </details>
    </section>
  );
}

export function Sidebar({ data, onSearch }: { data: SearchPageData; onSearch: (q: string) => void }) {
  const t = useT();
  const globals = data.globals;
  const hasInfobox = data.infoboxes.length > 0;
  // POST mode keeps the query out of the address bar, so the sidebar offers
  // the shareable URL reconstructed from the current search (upstream parity)
  const searchUrl = buildSearchUrl({
    q: data.q,
    categories: data.selected_categories.length > 0 ? data.selected_categories : undefined,
    pageno: data.pageno,
    language: data.current_language,
    time_range: data.time_range || undefined,
    timeout_limit: data.timeout_limit || undefined,
    safesearch: globals.safesearch,
  });

  return (
    <aside className="flex flex-col gap-3">
      {hasInfobox ? (
        <section aria-label={t("info")} className="flex flex-col gap-3">
          {data.infoboxes.map((infobox, index) => (
            <Infobox globals={globals} infobox={infobox} key={index} onSearch={onSearch} />
          ))}
        </section>
      ) : null}

      {globals.method === "POST" ? (
        <Box title={t("search_url")}>
          <ClickToCopy value={searchUrl}>
            <pre
              className="min-w-0 overflow-x-auto rounded-lg bg-surface-2 p-2 font-mono text-xs leading-relaxed break-all whitespace-pre-wrap text-ink-2"
              dir="ltr"
            >
              {searchUrl}
            </pre>
          </ClickToCopy>
        </Box>
      ) : null}
    </aside>
  );
}
