// SPDX-License-Identifier: Apache-2.0 WITH Commons-Clause-1.0

import { Link, Shell } from "../components/Shell.tsx";
import type { InfoPageData } from "../lib/types.ts";

export function InfoPage({ data, embedded = false }: { data: InfoPageData; embedded?: boolean }) {
  const globals = data.globals;
  return (
    <Shell embedded={embedded} globals={globals}>
      <main className="mx-auto w-full max-w-3xl flex-1 px-4 pb-16 sm:px-6">
        <nav aria-label="info pages" className="flex flex-wrap gap-1.5 py-5">
          {data.pages.map((page) => {
            const active = page.pagename === data.active_pagename;
            return (
              <Link
                className={`rounded-full border px-3.5 py-1.5 text-[13px] transition-colors ${
                  active
                    ? "border-accent bg-accent-soft font-medium text-accent"
                    : "border-line text-ink-2 hover:border-ink-3 hover:text-ink"
                }`}
                href={`/info/${page.locale}/${page.pagename}`}
                key={`${page.locale}/${page.pagename}`}
              >
                {page.title}
              </Link>
            );
          })}
        </nav>
        <article
          className="prose-basic animate-fade-up"
          dangerouslySetInnerHTML={{ __html: data.active_html }}
          dir="auto"
        />
      </main>
    </Shell>
  );
}
