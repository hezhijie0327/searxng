// SPDX-License-Identifier: Apache-2.0 WITH Commons-Clause-1.0

/**
 * Slide-in overlay panel: About / Stats / Preferences open as a right-side
 * drawer on top of the current page (URL unchanged, search state kept).
 * The panel fetches the target URL and renders the very same page component
 * that a direct visit would use.
 */

import { X } from "lucide-react";
import { createContext, type ReactNode, Suspense, useCallback, useContext, useEffect, useState } from "react";
import { InfoPage, PreferencesPage, StatsPage } from "../pages/lazyPages.ts";
import { useT } from "./i18n.ts";
import { extractPageData } from "./pageData.ts";
import type { AnyPageData } from "./types.ts";
import { isInfoPageData, isPreferencesPageData, isStatsPageData } from "./types.ts";

interface OverlayState {
  url: string;
  title: string;
  /** "document" panels fetch plain text (LICENSE.txt); "page" panels fetch page-data */
  mode: "page" | "document";
  data: AnyPageData | null;
  text: string | null;
  loading: boolean;
  error: string | null;
}

interface OverlayContextValue {
  openOverlay: (url: string, title: string) => void;
  /** open a plain-text document (LICENSE.txt ...) rendered inside the panel */
  openDocument: (title: string, url: string) => void;
  closeOverlay: () => void;
}

const OverlayContext = createContext<OverlayContextValue | null>(null);

export function useOverlay(): OverlayContextValue {
  const ctx = useContext(OverlayContext);
  if (!ctx) {
    throw new Error("useOverlay outside of OverlayProvider");
  }
  return ctx;
}

export function OverlayProvider({ children }: { children: ReactNode }) {
  const t = useT();
  const [state, setState] = useState<OverlayState | null>(null);

  const closeOverlay = useCallback(() => {
    setState(null);
  }, []);

  const openOverlay = useCallback((url: string, title: string) => {
    setState({ url, title, mode: "page", data: null, text: null, loading: true, error: null });
  }, []);

  const openDocument = useCallback((title: string, url: string) => {
    setState({ url, title, mode: "document", data: null, text: null, loading: true, error: null });
  }, []);

  // navigate the panel to another URL, keeping it open
  const openPanel = useCallback((url: string, title: string) => {
    setState((prev) => ({
      url,
      title: title || prev?.title || "",
      mode: "page",
      data: null,
      text: null,
      loading: true,
      error: null,
    }));
  }, []);

  // fetch the payload when a panel is requested
  useEffect(() => {
    if (!state?.loading) {
      return;
    }
    const controller = new AbortController();
    const isDocument = state.mode === "document";
    void fetch(state.url, { headers: { Accept: isDocument ? "text/plain" : "text/html" }, signal: controller.signal })
      .then(async (resp) => {
        if (!resp.ok) {
          throw new Error(`HTTP ${resp.status}`);
        }
        const body = await resp.text();
        if (isDocument) {
          setState((prev) => (prev && prev.url === state.url ? { ...prev, text: body, loading: false } : prev));
          return;
        }
        const data = extractPageData(body);
        setState((prev) => (prev && prev.url === state.url ? { ...prev, data, loading: false } : prev));
      })
      .catch((err) => {
        if (controller.signal.aborted) {
          return;
        }
        // not an app page (e.g. a static file) — fall back to a full load
        if (!isDocument && String(err).includes("page-data missing")) {
          window.location.assign(state.url);
          return;
        }
        setState((prev) => (prev && prev.url === state.url ? { ...prev, loading: false, error: String(err) } : prev));
      });
    return () => {
      controller.abort();
    };
  }, [state]);

  const onKeyDown = useCallback((event: globalThis.KeyboardEvent) => {
    if (event.key === "Escape") {
      setState(null);
    }
  }, []);

  useEffect(() => {
    if (!state) {
      return;
    }
    window.addEventListener("keydown", onKeyDown);
    return () => {
      window.removeEventListener("keydown", onKeyDown);
    };
  }, [state, onKeyDown]);

  return (
    <OverlayContext.Provider value={{ openOverlay, openDocument, closeOverlay }}>
      {children}
      {state ? (
        <div aria-label={state.title} aria-modal="true" className="fixed inset-0 z-50" role="dialog">
          <button
            aria-label={t("close")}
            className="absolute inset-0 cursor-default bg-black/60 animate-fade-in"
            onClick={closeOverlay}
            type="button"
          />
          <div className="absolute inset-y-0 end-0 flex w-full max-w-3xl flex-col bg-bg shadow-pop animate-slide-in-right">
            <div className="flex items-center justify-between border-b border-line px-5 py-3">
              <span className="text-sm font-semibold text-ink">{state.title}</span>
              <button
                aria-label={t("close")}
                className="grid size-9 place-items-center rounded-full text-ink-2 transition-colors hover:bg-surface-2 hover:text-ink"
                onClick={closeOverlay}
                type="button"
              >
                <X className="size-[18px]" />
              </button>
            </div>
            <div
              className="min-h-0 flex-1 overflow-y-auto pt-4"
              onClickCapture={(event) => {
                const anchor = (event.target as HTMLElement).closest("a");
                const href = anchor?.getAttribute("href");
                if (
                  !anchor ||
                  !href?.startsWith("/") ||
                  anchor.target === "_blank" ||
                  anchor.hasAttribute("download") ||
                  // /static/ holds files (LICENSE.txt...), not SPA pages
                  href.startsWith("/static/") ||
                  event.metaKey ||
                  event.ctrlKey ||
                  event.shiftKey ||
                  event.altKey
                ) {
                  return;
                }
                event.preventDefault();
                event.stopPropagation();
                openPanel(href, state?.title ?? "");
              }}
            >
              {state.loading ? (
                <div aria-busy="true" className="space-y-3 p-6">
                  {Array.from({ length: 6 }, (_, i) => (
                    <div className="zjs-skeleton h-12" key={i} />
                  ))}
                </div>
              ) : state.error ? (
                <p className="p-6 text-sm text-danger">{state.error}</p>
              ) : state.mode === "document" ? (
                <article className="px-5 pb-8">
                  <pre className="whitespace-pre-wrap break-words font-mono text-[11.5px] leading-relaxed text-ink-2">
                    {state.text}
                  </pre>
                </article>
              ) : state.data ? (
                <Suspense fallback={chunkFallback}>
                  <OverlayContent data={state.data} />
                </Suspense>
              ) : null}
            </div>
          </div>
        </div>
      ) : null}
    </OverlayContext.Provider>
  );
}

const chunkFallback = (
  <div aria-busy="true" className="space-y-3 p-6">
    {Array.from({ length: 6 }, (_, i) => (
      <div className="zjs-skeleton h-12" key={i} />
    ))}
  </div>
);

function OverlayContent({ data }: { data: AnyPageData }) {
  if (isPreferencesPageData(data)) {
    return <PreferencesPage data={data} embedded />;
  }
  if (isStatsPageData(data)) {
    return <StatsPage data={data} embedded />;
  }
  if (isInfoPageData(data)) {
    return <InfoPage data={data} embedded />;
  }
  return (
    <p className="p-6 text-sm text-ink-2">
      This page cannot be shown as a panel.{" "}
      <a className="text-accent" href={data.globals.about_url || "/"}>
        Open it here
      </a>
      .
    </p>
  );
}
