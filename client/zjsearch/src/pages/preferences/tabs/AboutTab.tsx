// SPDX-License-Identifier: Apache-2.0 WITH Commons-Clause-1.0

import { LoaderCircle } from "lucide-react";
import { useEffect, useState } from "react";
import { extractPageData } from "@/lib/pageData.ts";
import type { GlobalData, InfoPageData } from "@/lib/types.ts";
import { isInfoPageData } from "@/lib/types.ts";
import { InfoPage } from "@/pages/InfoPage.tsx";

/** 「信息」tab: the instance info pages (about / search syntax) rendered in
    place — fetched from the localized info entry point and reusing the
    embedded InfoPage (instance card, footer links and cross-page chips
    included). */
export function AboutTab({ globals }: { globals: GlobalData }) {
  const [data, setData] = useState<InfoPageData | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    void fetch(globals.about_url, { headers: { Accept: "text/html" }, signal: controller.signal })
      .then(async (resp) => {
        if (!resp.ok) {
          throw new Error(`HTTP ${resp.status}`);
        }
        const pageData = extractPageData(await resp.text());
        if (!isInfoPageData(pageData)) {
          throw new Error("unexpected page payload");
        }
        setData(pageData);
      })
      .catch((err) => {
        if (!controller.signal.aborted) {
          setError(String(err));
        }
      });
    return () => {
      controller.abort();
    };
  }, [globals.about_url]);

  if (error) {
    return <p className="p-6 text-sm text-danger">{error}</p>;
  }
  if (!data) {
    return (
      <div aria-busy="true" className="space-y-3">
        {Array.from({ length: 6 }, (_, i) => (
          <div className="zjs-skeleton h-12" key={i} />
        ))}
      </div>
    );
  }
  return <InfoPage data={data} embedded />;
}
