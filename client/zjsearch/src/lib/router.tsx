// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * Tiny SPA router: page loads come from the server shell, every in-app
 * navigation fetches the target URL and swaps the page payload (page-data)
 * via pushState.  On network failures we fall back to a full page load.
 */

import { createContext, useCallback, useContext, useEffect, useRef, useState } from "react";
import { extractPageData } from "./pageData.ts";
import type { AnyPageData } from "./types.ts";

export interface SearchParams {
  q: string;
  categories?: string[];
  pageno?: number;
  language?: string;
  time_range?: string;
  safesearch?: number;
  timeout_limit?: string;
  engine_data?: Record<string, Record<string, string>>;
}

export function buildSearchUrl(params: SearchParams): string {
  const query = new URLSearchParams();
  query.set("q", params.q);
  if (params.language) {
    query.set("language", params.language);
  }
  if (params.time_range) {
    query.set("time_range", params.time_range);
  }
  if (params.safesearch !== undefined) {
    query.set("safesearch", String(params.safesearch));
  }
  if (params.timeout_limit) {
    query.set("timeout_limit", params.timeout_limit);
  }
  if (params.pageno !== undefined && params.pageno > 1) {
    query.set("pageno", String(params.pageno));
  }
  if (params.categories && params.categories.length > 0) {
    query.set("categories", params.categories.join(","));
  }
  for (const [engine, kv] of Object.entries(params.engine_data ?? {})) {
    for (const [key, value] of Object.entries(kv)) {
      query.set(`engine_data-${engine}-${key}`, value);
    }
  }
  return `/search?${query.toString()}`;
}

export function parseSearchUrl(url: URL): SearchParams {
  const query = url.searchParams;
  const engine_data: Record<string, Record<string, string>> = {};
  for (const [key, value] of query.entries()) {
    if (key.startsWith("engine_data-")) {
      const rest = key.slice("engine_data-".length);
      const sep = rest.indexOf("-");
      if (sep > 0) {
        const engine = rest.slice(0, sep);
        const dataKey = rest.slice(sep + 1);
        if (!engine_data[engine]) {
          engine_data[engine] = {};
        }
        engine_data[engine][dataKey] = value;
      }
    }
  }
  const categories = query.get("categories");
  return {
    q: query.get("q") ?? "",
    categories: categories ? categories.split(",") : undefined,
    pageno: Number(query.get("pageno")) || 1,
    language: query.get("language") ?? undefined,
    time_range: query.get("time_range") ?? undefined,
    safesearch: query.has("safesearch") ? Number(query.get("safesearch")) : undefined,
    timeout_limit: query.get("timeout_limit") ?? undefined,
    engine_data: Object.keys(engine_data).length > 0 ? engine_data : undefined,
  };
}

export interface NavigateOptions {
  replace?: boolean;
  method?: "GET" | "POST";
  body?: FormData;
}

interface RouterContextValue {
  data: AnyPageData | null;
  loading: boolean;
  error: string | null;
  /** Navigate within the app; falls back to a full page load on failure. */
  navigate: (url: string, options?: NavigateOptions) => void;
  /** Convenience: navigate to /search with the given params. */
  search: (params: SearchParams, options?: NavigateOptions) => void;
  /** Re-fetch the current URL. */
  reload: () => void;
  href: string;
}

const RouterContext = createContext<RouterContextValue | null>(null);

function pageTitle(data: AnyPageData): string {
  const name = data.globals.instance_name;
  if (data.globals.page === "results" && "q" in data && data.q) {
    return `${data.q} - ${name}`;
  }
  if (data.globals.page === "index") {
    return name;
  }
  const labels: Record<string, string> = {
    preferences: "Preferences",
    stats: "Engine stats",
    info: "Info",
    "404": "Page not found",
  };
  return `${labels[data.globals.page] ?? name} - ${name}`;
}

export function RouterProvider({
  initialData,
  children,
}: {
  initialData: AnyPageData | null;
  children: React.ReactNode;
}) {
  const [data, setData] = useState<AnyPageData | null>(initialData);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [href, setHref] = useState(() => window.location.href);
  const abortRef = useRef<AbortController | null>(null);
  const seqRef = useRef(0);

  const load = useCallback(
    async (url: string, options: NavigateOptions = {}, historyMode: "push" | "replace" | "none" = "push") => {
      const seq = ++seqRef.current;
      abortRef.current?.abort();
      const controller = new AbortController();
      abortRef.current = controller;
      setLoading(true);
      setError(null);

      try {
        const resp = await fetch(url, {
          method: options.method ?? "GET",
          body: options.body,
          signal: controller.signal,
          headers: { Accept: "text/html" },
          redirect: "follow",
        });
        if (!resp.ok) {
          throw new Error(`HTTP ${resp.status}`);
        }
        const html = await resp.text();
        const pageData = extractPageData(html);
        if (seq !== seqRef.current) {
          return; // superseded by a newer navigation
        }
        // the final URL after redirects (e.g. POST /preferences -> /)
        const finalUrl = new URL(resp.url, window.location.href).href;
        if (historyMode !== "none") {
          const historyMethod = historyMode === "replace" ? "replaceState" : "pushState";
          // keep the payload out of the history state — the page data already
          // lives in React state and popstate re-fetches by URL, so storing
          // it here would only duplicate memory for every visited page
          window.history[historyMethod](null, "", finalUrl);
        }
        setHref(finalUrl);
        setData(pageData);
        setLoading(false);
        document.title = pageTitle(pageData);
        window.scrollTo(0, 0);
      } catch (err) {
        if (controller.signal.aborted || seq !== seqRef.current) {
          return;
        }
        // network/CORS failure (e.g. external-bang redirect): full page load
        if (err instanceof TypeError) {
          window.location.assign(url);
          return;
        }
        setError(err instanceof Error ? err.message : String(err));
        setLoading(false);
      }
    },
    [],
  );

  const navigate = useCallback(
    (url: string, options?: NavigateOptions) => {
      if (url === window.location.href && !options?.body) {
        return;
      }
      void load(url, options, options?.replace ? "replace" : "push");
    },
    [load],
  );

  const search = useCallback(
    (params: SearchParams, options?: NavigateOptions) => {
      navigate(buildSearchUrl(params), options);
    },
    [navigate],
  );

  const reload = useCallback(() => {
    void load(window.location.href, {}, "none");
  }, [load]);

  const hrefRef = useRef(href);
  hrefRef.current = href;

  useEffect(() => {
    const onPopState = () => {
      // ignore hash-only changes (image viewer open/close etc.)
      const stripHash = (value: string) => {
        const url = new URL(value);
        url.hash = "";
        return url.href;
      };
      if (stripHash(window.location.href) === stripHash(hrefRef.current)) {
        return;
      }
      void load(window.location.href, {}, "none");
    };
    window.addEventListener("popstate", onPopState);
    return () => {
      window.removeEventListener("popstate", onPopState);
      abortRef.current?.abort();
    };
  }, [load]);

  return (
    <RouterContext.Provider value={{ data, loading, error, navigate, search, reload, href }}>
      {children}
    </RouterContext.Provider>
  );
}

export function useRouter(): RouterContextValue {
  const ctx = useContext(RouterContext);
  if (!ctx) {
    throw new Error("useRouter outside of RouterProvider");
  }
  return ctx;
}
