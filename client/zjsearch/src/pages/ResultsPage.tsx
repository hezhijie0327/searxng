// SPDX-License-Identifier: AGPL-3.0-or-later

import { useEffect, useMemo, useRef, useState } from "react";
import { HelpModal } from "../components/HelpModal.tsx";
import { ArrowUpIcon, InfoIcon } from "../components/icons.tsx";
import { Answers } from "../components/results/Answers.tsx";
import { NewsCard, ProductGrid, ResultCard, ResultSkeleton, VideoGrid } from "../components/results/cards.tsx";
import { ImageGrid } from "../components/results/ImageGrid.tsx";
import { Pagination } from "../components/results/Pagination.tsx";
import { Sidebar } from "../components/results/Sidebar.tsx";
import { SearchBox } from "../components/SearchBox.tsx";
import { CategoryTabs, type FilterValues, SearchFilters } from "../components/SearchControls.tsx";
import { HeaderActions, Link, Shell } from "../components/Shell.tsx";
import { tryEvaluateExpression } from "../features/calculator.ts";
import { useHotkeys } from "../features/hotkeys.ts";
import { useT } from "../lib/i18n.ts";
import { extractPageData } from "../lib/pageData.ts";
import { parseSearchUrl, useRouter } from "../lib/router.tsx";
import { useHasPlugin, useSettings } from "../lib/settings.ts";
import type { ResultItem, SearchPageData } from "../lib/types.ts";

function BackToTop() {
  const t = useT();
  const [visible, setVisible] = useState(false);
  useEffect(() => {
    const onScroll = () => {
      setVisible(window.scrollY > 400);
    };
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => {
      window.removeEventListener("scroll", onScroll);
    };
  }, []);
  if (!visible) {
    return null;
  }
  return (
    <button
      aria-label={t("back_to_top")}
      className="fixed bottom-6 right-6 z-40 grid size-11 place-items-center rounded-full border border-line bg-surface text-ink-2 shadow-pop transition-colors hover:text-accent animate-fade-in"
      onClick={() => {
        window.scrollTo({ top: 0, behavior: "smooth" });
      }}
      type="button"
    >
      <ArrowUpIcon className="size-5" />
    </button>
  );
}

function NoResults({ pageno }: { pageno: number }) {
  const t = useT();
  const firstPage = pageno === 1;
  return (
    <div className="mx-auto max-w-md rounded-2xl border border-line bg-surface p-6 text-sm text-ink-2 animate-fade-up">
      <p className="flex items-center gap-2 font-medium text-ink">
        <InfoIcon className="size-4 text-accent" />
        {firstPage ? t("sorry") : ""}
      </p>
      <p className="mt-2">{firstPage ? t("no_results_found") : t("no_more_results")}</p>
      <ul className="mt-2 list-disc space-y-1 pl-5">
        {firstPage ? (
          <>
            <li>
              <button className="text-accent hover:underline" onClick={() => window.location.reload()} type="button">
                {t("refresh_page")}
              </button>
            </li>
            <li>{t("try_other_query")}</li>
            <li>
              {t("change_engines_prefs")}{" "}
              <Link className="text-accent hover:underline" href="/preferences">
                /preferences
              </Link>
            </li>
            <li>
              {t("switch_instance")}{" "}
              <a className="text-accent hover:underline" href="https://searx.space" rel="noreferrer" target="_blank">
                https://searx.space
              </a>
            </li>
          </>
        ) : (
          <li>{t("go_previous_page")}</li>
        )}
      </ul>
    </div>
  );
}

function Corrections({ data, onSearch }: { data: SearchPageData; onSearch: (q: string) => void }) {
  const t = useT();
  if (data.corrections.length === 0) {
    return null;
  }
  return (
    <div className="flex flex-wrap items-center gap-2 text-sm">
      <span className="text-ink-3">{t("try_searching_for")}</span>
      {data.corrections.map((correction) => (
        <button
          className="rounded-full bg-accent-soft px-3 py-1 font-medium text-accent transition-colors hover:bg-accent-strong hover:text-accent-contrast"
          dir="auto"
          key={correction.q}
          onClick={() => {
            onSearch(correction.q);
          }}
          type="button"
        >
          {correction.title}
        </button>
      ))}
    </div>
  );
}

/** Consecutive same-template results form groups (image strips in mixed mode). */
function groupResults(
  results: ResultItem[],
): Array<{ template: string; items: Array<{ result: ResultItem; index: number }> }> {
  const groups: Array<{ template: string; items: Array<{ result: ResultItem; index: number }> }> = [];
  for (let index = 0; index < results.length; index += 1) {
    const result = results[index];
    const template = result?.template ?? "default";
    const last = groups[groups.length - 1];
    if (last && last.template === template) {
      last.items.push({ result: result as ResultItem, index });
    } else {
      groups.push({ template, items: [{ result: result as ResultItem, index }] });
    }
  }
  return groups;
}

function InfiniteScrollSentinel({ onNext, error, loading }: { onNext: () => void; error: boolean; loading: boolean }) {
  const ref = useRef<HTMLDivElement>(null);
  const t = useT();
  useEffect(() => {
    const el = ref.current;
    if (!el) {
      return;
    }
    const observer = new IntersectionObserver(
      (entries) => {
        if (entries.some((entry) => entry.isIntersecting)) {
          onNext();
        }
      },
      { rootMargin: "320px" },
    );
    observer.observe(el);
    return () => {
      observer.disconnect();
    };
  }, [onNext]);

  if (error) {
    return <p className="py-4 text-center text-sm text-danger">{t("error_loading_next_page")}</p>;
  }
  return (
    <div aria-busy={loading} className="flex justify-center py-6" ref={ref}>
      <div className="size-6 animate-spin-slow rounded-full border-2 border-line border-t-accent-strong" />
    </div>
  );
}

export function ResultsPage({ data }: { data: SearchPageData }) {
  const t = useT();
  const { search, loading, error, href } = useRouter();
  const hasPlugin = useHasPlugin();
  const infiniteScroll = hasPlugin("infiniteScroll");

  const globals = data.globals;
  const [selectedCategories, setSelectedCategories] = useState<string[]>(
    data.selected_categories.length > 0 ? data.selected_categories : [globals.default_category],
  );
  useEffect(() => {
    setSelectedCategories(data.selected_categories.length > 0 ? data.selected_categories : [globals.default_category]);
  }, [data, globals.default_category]);

  // the URL is the source of truth for the active filter values
  // biome-ignore lint/correctness/useExhaustiveDependencies: URL is the source of truth
  const urlParams = useMemo(() => {
    try {
      return parseSearchUrl(new URL(window.location.href));
    } catch {
      return null;
    }
  }, [href]);

  const [filterValues, setFilterValues] = useState<FilterValues>(() => ({
    language: urlParams?.language ?? data.current_language ?? globals.language,
    time_range: urlParams?.time_range ?? data.time_range ?? "",
    safesearch: urlParams?.safesearch ?? globals.safesearch,
    search_language: data.search_language,
  }));

  // re-sync filters after any navigation (back/forward, payload change)
  // biome-ignore lint/correctness/useExhaustiveDependencies: URL is the source of truth
  useEffect(() => {
    setFilterValues({
      language: urlParams?.language ?? data.current_language ?? globals.language,
      time_range: urlParams?.time_range ?? data.time_range ?? "",
      safesearch: urlParams?.safesearch ?? globals.safesearch,
      search_language: data.search_language,
    });
  }, [href]);

  const settings = useSettings();
  const [helpOpen, setHelpOpen] = useState(false);
  const [hotkeysSelected, setHotkeysSelected] = useState(-1);
  const listRef = useRef<HTMLDivElement | null>(null);
  const [appended, setAppended] = useState<ResultItem[]>([]);
  const [appendState, setAppendState] = useState<"idle" | "loading" | "error" | "done">("idle");
  const appendedHref = useRef(href);

  useEffect(() => {
    if (appendedHref.current !== href) {
      appendedHref.current = href;
      setAppended([]);
      setAppendState("idle");
    }
  }, [href]);

  const buildParams = (
    overrides?: Partial<{
      q: string;
      categories: string[];
      pageno: number;
      language: string;
      time_range: string;
      safesearch: number;
    }>,
  ) => ({
    q: overrides?.q ?? data.q,
    categories: overrides?.categories ?? selectedCategories,
    pageno: overrides?.pageno ?? data.pageno,
    language: overrides?.language ?? filterValues.language,
    time_range: overrides?.time_range ?? filterValues.time_range,
    safesearch: overrides?.safesearch ?? filterValues.safesearch,
    timeout_limit: data.timeout_limit || undefined,
    engine_data: data.engine_data && Object.keys(data.engine_data).length > 0 ? data.engine_data : undefined,
  });

  const submitQuery = (q: string) => {
    search(buildParams({ q, pageno: 1 }));
  };

  const onSearchCategories = (categories: string[]) => {
    setSelectedCategories(categories);
    search(buildParams({ categories, pageno: 1 }));
  };

  const onFilters = (next: Partial<FilterValues>) => {
    setFilterValues((prev) => ({ ...prev, ...next }));
    search(
      buildParams({
        language: next.language ?? filterValues.language,
        time_range: next.time_range ?? filterValues.time_range,
        safesearch: next.safesearch ?? filterValues.safesearch,
        pageno: 1,
      }),
    );
  };

  const onPage = (pageno: number) => {
    search(buildParams({ pageno }));
  };
  const onPageRef = useRef(onPage);
  onPageRef.current = onPage;

  const loadNextPage = useRef(() => {});
  loadNextPage.current = () => {
    if (appendState !== "idle") {
      return;
    }
    setAppendState("loading");
    const nextUrl = (() => {
      const params = new URLSearchParams();
      params.set("q", data.q);
      params.set("pageno", String(data.pageno + 1));
      if (filterValues.language) {
        params.set("language", filterValues.language);
      }
      if (filterValues.time_range) {
        params.set("time_range", filterValues.time_range);
      }
      params.set("safesearch", String(filterValues.safesearch));
      if (selectedCategories.length > 0) {
        params.set("categories", selectedCategories.join(","));
      }
      for (const [engine, kv] of Object.entries(data.engine_data ?? {})) {
        for (const [key, value] of Object.entries(kv)) {
          params.set(`engine_data-${engine}-${key}`, value);
        }
      }
      return `/search?${params.toString()}`;
    })();
    void fetch(nextUrl, { headers: { Accept: "text/html" } })
      .then(async (resp) => {
        if (!resp.ok) {
          throw new Error(`HTTP ${resp.status}`);
        }
        const next = extractPageData(await resp.text()) as SearchPageData;
        setAppended((prev) => [...prev, ...next.results]);
        setAppendState(next.paging ? "idle" : "done");
      })
      .catch(() => {
        setAppendState("error");
      });
  };

  // ----- keyboard navigation (default / vim layouts) -----
  const selectedCard = () => {
    const cards = listRef.current ? Array.from(listRef.current.querySelectorAll("article")) : [];
    return hotkeysSelected >= 0 ? (cards[hotkeysSelected] as HTMLElement | undefined) : undefined;
  };
  const hotkeyTarget = {
    move: (delta: number) => {
      const cards = listRef.current ? Array.from(listRef.current.querySelectorAll("article")) : [];
      if (cards.length === 0) {
        return;
      }
      setHotkeysSelected((prev) => {
        const next = Math.min(cards.length - 1, Math.max(0, prev + delta));
        cards[next]?.scrollIntoView({ block: "center", behavior: "smooth" });
        return next;
      });
    },
    open: (newTab: boolean) => {
      const href = selectedCard()?.querySelector("a[href]")?.getAttribute("href");
      if (href) {
        if (newTab) {
          window.open(href, "_blank", "noopener");
        } else {
          window.location.assign(href);
        }
      }
    },
    yank: () => selectedCard()?.querySelector("a[href]")?.getAttribute("href") ?? null,
    page: (delta: number) => {
      const next = data.pageno + delta;
      if (next >= 1 && (delta < 0 || data.paging)) {
        onPageRef.current(next);
      }
    },
    focusSearch: () => {
      (document.querySelector('input[name="q"]') as HTMLInputElement | null)?.focus();
    },
  };
  useHotkeys(settings.hotkeys, hotkeyTarget, () => {
    setHelpOpen(true);
  });

  const allResults = useMemo(() => [...data.results, ...appended], [data.results, appended]);

  // Page-level layout intent (Kagi-style per-category presentation): a single
  // selected category signals intent, `only_template` additionally catches
  // bang-limited searches where every result shares one template.
  const singleCategory = selectedCategories.length === 1 ? selectedCategories[0] : null;
  const isImagePage =
    (data.only_template === "images" || singleCategory === "images") &&
    allResults.every((result) => result.template === "images" || result.thumbnail_src || result.img_src);
  const isVideoPage = data.only_template === "videos" || singleCategory === "videos";
  const isProductPage = (data.only_template === "products" || singleCategory === "products") && !isVideoPage;
  const isNewsPage = singleCategory === "news" && !isImagePage && !isVideoPage;
  const isMapPage = singleCategory === "map" && !isImagePage && !isVideoPage;
  const isMusicPage = singleCategory === "music" && !isImagePage && !isVideoPage;

  // biome-ignore lint/correctness/useExhaustiveDependencies: href is the trigger
  useEffect(() => {
    setHotkeysSelected(-1);
  }, [href]);

  // client-side calculator answer (server plugin "calculator" enabled)
  const calcAnswer = useMemo(() => {
    if (!hasPlugin("calculator")) {
      return null;
    }
    const calc = tryEvaluateExpression(data.q);
    return calc ? ({ template: "answer/legacy.html", answer: `${calc.expr} = ${calc.value}`, url: "" } as const) : null;
  }, [data.q, hasPlugin]);
  const showSkeletons = loading && !error;

  return (
    <Shell globals={globals} hideTopNav>
      <header className="border-b border-line">
        <div className="zjs-results-header-row mx-auto flex w-full items-center gap-4 px-4 pt-3 sm:px-6">
          {/* brand mark only - no home link needed, everything opens as a drawer;
              hidden on small screens so the query box keeps enough width */}
          <span className="hidden min-[480px]:block shrink-0 select-none text-xl font-extrabold tracking-tight text-ink">
            {globals.instance_name}
            <span className="text-accent-strong">.</span>
          </span>
          <div className="min-w-0 flex-1">
            <SearchBox initialQuery={data.q} onSubmitQuery={submitQuery} />
          </div>
          <HeaderActions globals={globals} />
        </div>
        <div className="zjs-results-header-row mx-auto px-4 pt-1 sm:px-6">
          <CategoryTabs globals={globals} onSearch={onSearchCategories} selected={selectedCategories} />
        </div>
        <div className="zjs-results-header-row mx-auto px-4 pb-1 sm:px-6">
          <SearchFilters globals={globals} onChange={onFilters} values={filterValues} />
        </div>
      </header>

      <main className="zjs-results-main mx-auto w-full flex-1 px-4 sm:px-6">
        <div className="flex flex-col gap-6 lg:flex-row lg:gap-8">
          <div className="min-w-0 flex-1 pt-4">
            {error ? (
              <div
                className="mb-4 rounded-2xl border border-danger/30 bg-danger/10 p-4 text-sm text-danger"
                role="alert"
              >
                {t("error_loading_next_page")} ({error})
              </div>
            ) : null}

            {showSkeletons ? (
              <div aria-busy="true">
                {Array.from({ length: 5 }, (_, index) => (
                  <ResultSkeleton key={index} />
                ))}
              </div>
            ) : (
              <>
                <Corrections data={data} onSearch={submitQuery} />
                <div className="mt-3 space-y-3">
                  <Answers answers={calcAnswer ? [calcAnswer, ...data.answers] : data.answers} />
                </div>

                {allResults.length === 0 && data.answers.length === 0 ? (
                  <div className="mt-6">
                    <NoResults pageno={data.pageno} />
                  </div>
                ) : isImagePage ? (
                  <div className="mt-4">
                    <ImageGrid results={allResults} />
                  </div>
                ) : isVideoPage ? (
                  <div className="mt-4">
                    <VideoGrid globals={globals} results={allResults} />
                  </div>
                ) : isMusicPage ? (
                  <div className="mt-2 space-y-1">
                    {allResults.map((result, index) => (
                      <div
                        className={`animate-fade-up rounded-2xl ${
                          index === hotkeysSelected ? "bg-surface ring-1 ring-accent-strong" : ""
                        }`}
                        key={index}
                        style={{ animationDelay: `${Math.min(index * 30, 300)}ms` }}
                      >
                        <ResultCard globals={globals} mediaOpen result={result} />
                      </div>
                    ))}
                  </div>
                ) : isProductPage ? (
                  <div className="mt-4">
                    <ProductGrid globals={globals} results={allResults} />
                  </div>
                ) : isNewsPage ? (
                  <div className="mt-2 space-y-1">
                    {allResults.map((result, index) => (
                      <div
                        className={`animate-fade-up rounded-2xl ${
                          index === hotkeysSelected ? "bg-surface ring-1 ring-accent-strong" : ""
                        }`}
                        key={index}
                        style={{ animationDelay: `${Math.min(index * 30, 300)}ms` }}
                      >
                        <NewsCard globals={globals} result={result} />
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="mt-2" ref={listRef}>
                    {groupResults(allResults).map((group, groupIndex) =>
                      group.template === "images" ? (
                        <div className="grid grid-cols-2 gap-3 py-2 sm:grid-cols-3 lg:grid-cols-4" key={groupIndex}>
                          {group.items.map(({ result, index }) => (
                            <ResultCard
                              globals={globals}
                              key={index}
                              onOpenImage={() => window.open(result.img_src ?? result.url, "_blank", "noopener")}
                              result={result}
                            />
                          ))}
                        </div>
                      ) : (
                        <div key={groupIndex}>
                          {group.items.map(({ result, index }) => (
                            <div
                              className={`animate-fade-up rounded-2xl ${
                                index === hotkeysSelected ? "bg-surface ring-1 ring-accent-strong" : ""
                              }`}
                              key={index}
                              style={{ animationDelay: `${Math.min(index * 30, 300)}ms` }}
                            >
                              <ResultCard autoOpenMap={isMapPage} globals={globals} result={result} />
                            </div>
                          ))}
                        </div>
                      ),
                    )}
                  </div>
                )}

                {infiniteScroll && appendState !== "done" && (data.paging || appended.length > 0) ? (
                  <InfiniteScrollSentinel
                    error={appendState === "error"}
                    loading={appendState === "loading"}
                    onNext={() => {
                      loadNextPage.current();
                    }}
                  />
                ) : (
                  <Pagination onPage={onPage} pageno={data.pageno} paging={data.paging} />
                )}
              </>
            )}
          </div>

          {!isImagePage && !isVideoPage && !isProductPage ? (
            <div className="w-full shrink-0 pt-4 lg:w-80 lg:max-h-[calc(100dvh-9rem)] lg:self-start lg:overflow-y-auto lg:pb-6 [scrollbar-width:thin]">
              <Sidebar data={data} onSearch={submitQuery} />
            </div>
          ) : null}
        </div>
      </main>

      <BackToTop />
      {helpOpen ? <HelpModal layout={settings.hotkeys} onClose={() => setHelpOpen(false)} /> : null}
    </Shell>
  );
}
