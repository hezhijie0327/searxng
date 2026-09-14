// SPDX-License-Identifier: Apache-2.0 WITH Commons-Clause-1.0

import { Info, RefreshCw, Search } from "lucide-react";
import { useT } from "@/lib/i18n.ts";
import type { SearchPageData } from "@/lib/types.ts";

export function NoResults({ pageno, hasInfobox }: { pageno: number; hasInfobox: boolean }) {
  const t = useT();
  const firstPage = pageno === 1;
  if (hasInfobox && firstPage) {
    return (
      <div className="rounded-2xl border border-line bg-surface p-4 text-sm text-ink-2">
        <p className="flex items-center gap-2">
          <Info className="size-4 shrink-0 text-accent" />
          {t("no_web_results")}
        </p>
      </div>
    );
  }
  return (
    <div className="mx-auto max-w-md rounded-2xl border border-line bg-surface p-6 text-sm text-ink-2 animate-fade-up">
      <p className="flex items-center gap-2 font-medium text-ink">
        <Info className="size-4 text-accent" />
        {firstPage ? t("sorry") : ""}
      </p>
      <p className="mt-2">{firstPage ? t("no_results_found") : t("no_more_results")}</p>
      <ul className="mt-2 list-disc space-y-1 pl-5">
        {firstPage ? (
          <>
            <li>
              <button
                className="inline-flex items-center gap-1 text-accent hover:underline"
                onClick={() => window.location.reload()}
                type="button"
              >
                <RefreshCw className="size-3.5" />
                {t("refresh_page")}
              </button>
            </li>
            <li>{t("try_other_query")}</li>
          </>
        ) : (
          <li>{t("go_previous_page")}</li>
        )}
      </ul>
    </div>
  );
}

export function Corrections({ data, onSearch }: { data: SearchPageData; onSearch: (q: string) => void }) {
  const t = useT();
  if (data.corrections.length === 0) {
    return null;
  }
  return (
    <div className="flex flex-wrap items-center gap-2 text-sm">
      <span className="text-ink-3">{t("try_searching_for")}</span>
      {data.corrections.map((correction) => (
        <button
          className="rounded-full bg-accent-soft px-2 py-0.5 font-medium text-accent transition-colors hover:bg-accent-strong hover:text-accent-contrast"
          dir="auto"
          key={correction.q}
          onClick={() => {
            onSearch(correction.q);
          }}
          type="button"
        >
          <Search className="size-3 shrink-0" />
          {correction.title}
        </button>
      ))}
    </div>
  );
}
