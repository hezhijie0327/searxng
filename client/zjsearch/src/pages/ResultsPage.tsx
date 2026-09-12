// SPDX-License-Identifier: AGPL-3.0-or-later

import { useEffect, useMemo, useRef, useState } from "react";
import { BackToTop } from "../components/BackToTop.tsx";
import { HelpModal } from "../components/HelpModal.tsx";
import { Answers, CalculatorAnswer } from "../components/results/Answers.tsx";
import { collectBlocks } from "../components/results/blocks.ts";
import { ResultSkeleton } from "../components/results/cardParts.tsx";
import { DictionaryCard, NewsCard, PaperCard, ResultCard } from "../components/results/cards.tsx";
import { Corrections, NoResults } from "../components/results/EmptyStates.tsx";
import { FilesGrid } from "../components/results/FilesGrid.tsx";
import { GroupHeader } from "../components/results/GroupHeader.tsx";
import { AppsGrid, PosterGrid, ProductGrid, VideoGrid } from "../components/results/grids.tsx";
import { ImageGrid } from "../components/results/ImageGrid.tsx";
import { InfiniteScrollSentinel } from "../components/results/InfiniteScroll.tsx";
import { MusicGrid } from "../components/results/MusicGrid.tsx";
import { PackageGrid } from "../components/results/PackageGrid.tsx";
import { Pagination } from "../components/results/Pagination.tsx";
import { DebugPanels, Infobox, Sidebar, SuggestionsBox } from "../components/results/Sidebar.tsx";
import { SearchBox } from "../components/SearchBox.tsx";
import { CategoryTabs, type FilterValues, SearchFilters } from "../components/SearchControls.tsx";
import { HeaderActions, Link, Shell } from "../components/Shell.tsx";
import { tryEvaluateExpression } from "../features/calculator.ts";
import { useHotkeys } from "../features/hotkeys.ts";
import { useT } from "../lib/i18n.ts";
import { extractPageData } from "../lib/pageData.ts";
import { buildSearchUrl, parseSearchUrl, useRouter } from "../lib/router.tsx";
import { useHasPlugin, useSettings } from "../lib/settings.ts";
import type { ResultItem, SearchPageData } from "../lib/types.ts";

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
  const [collapsedBlocks, setCollapsedBlocks] = useState<Record<string, boolean>>({});
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
    // a bang search pins its resolved category into selectedCategories; a
    // plain follow-up query must fall back to the user's default category
    // (the preferences cookie) instead of silently keeping the bang's
    const previousWasBang = data.q.trim().startsWith("!");
    const nextIsBang = q.trim().startsWith("!");
    const cookieDefault = document.cookie
      .match(/(?:^|; *)categories=([^;]*)/)?.[1]
      ?.split(",")
      .filter(Boolean);
    const categories =
      previousWasBang && !nextIsBang
        ? cookieDefault && cookieDefault.length > 0
          ? cookieDefault
          : [globals.default_category]
        : selectedCategories;
    search(buildParams({ q, pageno: 1, categories }));
  };

  const onSearchCategories = (categories: string[]) => {
    setSelectedCategories(categories);
    search(buildParams({ categories, pageno: 1 }));
  };

  const onFilters = (next: Partial<FilterValues>) => {
    // filters apply on the next search submit (magnifier / Enter) — switching
    // one alone must not fire a new search
    setFilterValues((prev) => ({ ...prev, ...next }));
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
    const nextUrl = buildSearchUrl(buildParams({ pageno: data.pageno + 1 }));
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
  // navigable cards are marked with data-hotkey-index (the index into
  // allResults); grids without per-result cards (images) are skipped
  const selectedCard = () => {
    if (hotkeysSelected < 0 || !listRef.current) {
      return undefined;
    }
    return listRef.current.querySelector<HTMLElement>(`[data-hotkey-index="${hotkeysSelected}"]`) ?? undefined;
  };
  const hotkeyTarget = {
    move: (delta: number) => {
      const cards = listRef.current
        ? Array.from(listRef.current.querySelectorAll<HTMLElement>("[data-hotkey-index]"))
        : [];
      if (cards.length === 0) {
        return;
      }
      const pos =
        hotkeysSelected < 0 ? -1 : cards.findIndex((card) => card.dataset.hotkeyIndex === String(hotkeysSelected));
      const next = cards[Math.min(cards.length - 1, Math.max(0, pos + delta))];
      if (!next) {
        return;
      }
      next.scrollIntoView({ block: "center", behavior: "smooth" });
      setHotkeysSelected(Number(next.dataset.hotkeyIndex));
    },
    open: (newTab: boolean) => {
      const href = selectedCard()?.querySelector("a[href]")?.getAttribute("href");
      if (href) {
        // o/Enter follows the "results in new tabs" preference, t/v forces it
        if (newTab || globals.results_on_new_tab) {
          window.open(href, "_blank", "noopener,noreferrer");
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
    setHelpOpen((open) => !open);
  });

  const allResults = useMemo(() => [...data.results, ...appended], [data.results, appended]);

  // Page-level layout intent (Kagi-style per-category presentation): a single
  // selected category signals intent, `only_template` additionally catches
  // bang-limited searches where every result shares one template.
  const singleCategory = selectedCategories.length === 1 ? selectedCategories[0] : null;
  // engine bangs (`!imdb bat`) run with the pseudo category "none"; every
  // result still carries its real category, so a bang search whose results
  // all agree on one category inherits that category's presentation
  const firstResult = allResults[0];
  const bangCategory =
    singleCategory === "none" &&
    firstResult !== undefined &&
    allResults.every((result) => result.category === firstResult.category)
      ? firstResult.category
      : null;
  const isImagePage =
    (data.only_template === "images" || singleCategory === "images" || bangCategory === "images") &&
    allResults.every((result) => result.template === "images" || result.thumbnail_src || result.img_src);
  const isVideoPage = data.only_template === "videos" || singleCategory === "videos" || bangCategory === "videos";
  const isProductPage =
    (data.only_template === "products" || singleCategory === "products" || bangCategory === "products") && !isVideoPage;
  const isNewsPage = (singleCategory === "news" || bangCategory === "news") && !isImagePage && !isVideoPage;
  const isMapPage = (singleCategory === "map" || bangCategory === "map") && !isImagePage && !isVideoPage;
  const isMusicPage = (singleCategory === "music" || bangCategory === "music") && !isImagePage && !isVideoPage;
  const isMoviesPage = (singleCategory === "movies" || bangCategory === "movies") && !isImagePage && !isVideoPage;
  const isDictionaryPage =
    singleCategory === "dictionaries" ||
    singleCategory === "define" ||
    bangCategory === "dictionaries" ||
    bangCategory === "define";
  const isAppsPage = singleCategory === "apps" || bangCategory === "apps";
  const isPackagesPage = singleCategory === "packages" || bangCategory === "packages";
  // science intent renders every result in the scholarly layout; a
  // paper-only bang search (`!pubmed ...`) gets the same treatment
  const isSciencePage =
    (singleCategory === "science" || bangCategory === "science" || data.only_template === "paper") &&
    !isImagePage &&
    !isVideoPage &&
    !isMusicPage;
  // files intent (or a torrent-only bang search) gets the file-tile grid;
  // torrents keep the transfer card inside mixed searches
  const isFilesPage =
    (singleCategory === "files" || bangCategory === "files" || data.only_template === "torrent") &&
    !isImagePage &&
    !isVideoPage &&
    !isMusicPage &&
    !isSciencePage;

  // biome-ignore lint/correctness/useExhaustiveDependencies: href is the trigger
  useEffect(() => {
    setHotkeysSelected(-1);
  }, [href]);

  // client-side calculator answer (server plugin "calculator" enabled)
  const calc = useMemo(() => {
    if (!hasPlugin("calculator")) {
      return null;
    }
    return tryEvaluateExpression(data.q);
  }, [data.q, hasPlugin]);
  const showSkeletons = loading && !error;

  return (
    <Shell globals={globals} hideTopNav>
      <header>
        <div className="zjs-results-header-row mx-auto flex w-full items-center gap-4 px-4 pt-3 sm:px-6">
          {/* brand links back to the home page (SPA navigation);
              hidden on small screens so the query box keeps enough width */}
          <Link
            ariaLabel={globals.instance_name}
            className="hidden min-[480px]:block shrink-0 select-none text-xl font-extrabold tracking-tight text-ink"
            href="/"
            title={globals.instance_name}
          >
            {globals.instance_name}
            <span className="text-accent-strong">.</span>
          </Link>
          <div className="min-w-0 flex-1 max-w-2xl">
            <SearchBox initialQuery={data.q} onSubmitQuery={submitQuery} />
          </div>
          <div className="ms-auto">
            <HeaderActions globals={globals} />
          </div>
        </div>
      </header>

      <main className="zjs-results-main mx-auto w-full flex-1 px-4 sm:px-6">
        <div className="flex flex-col gap-6 lg:flex-row lg:gap-8">
          <div className="min-w-0 flex-1 pt-4" ref={listRef}>
            {/* Kagi layout: the tabs and filters live in the results column so
                the infobox sidebar rises to the top of the page */}
            <CategoryTabs
              globals={globals}
              onSearch={onSearchCategories}
              onSelectionChange={setSelectedCategories}
              selected={selectedCategories}
            />
            <div className="mt-1">
              <SearchFilters globals={globals} onChange={onFilters} values={filterValues} />
            </div>
            {!showSkeletons && !error ? (
              <>
                <div className="mt-2">
                  <DebugPanels
                    data={data}
                    leading={
                      <span>
                        {t("meta_found")} {allResults.length} {t("meta_results")}
                      </span>
                    }
                  />
                </div>
                <div className="mt-3.5">
                  <SuggestionsBox data={data} onSearch={submitQuery} />
                </div>
              </>
            ) : null}

            {!showSkeletons && data.infoboxes.length > 0 ? (
              <div className="mt-3 flex flex-col gap-3 lg:hidden">
                {data.infoboxes.map((infobox, index) => (
                  <Infobox globals={globals} infobox={infobox} key={index} onSearch={submitQuery} />
                ))}
              </div>
            ) : null}
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
                  {calc ? <CalculatorAnswer calc={calc} /> : null}
                  <Answers answers={data.answers} query={data.q} />
                </div>

                {allResults.length === 0 && data.answers.length === 0 ? (
                  <div className="mt-6">
                    <NoResults hasInfobox={data.infoboxes.length > 0} pageno={data.pageno} />
                  </div>
                ) : isImagePage ? (
                  <div className="mt-4">
                    <ImageGrid results={allResults} />
                  </div>
                ) : isVideoPage ? (
                  <div className="mt-4">
                    <VideoGrid globals={globals} results={allResults} selected={hotkeysSelected} />
                  </div>
                ) : isMusicPage ? (
                  <div className="mt-4">
                    <MusicGrid globals={globals} results={allResults} selected={hotkeysSelected} />
                  </div>
                ) : isMoviesPage ? (
                  <div className="mt-4">
                    <PosterGrid globals={globals} results={allResults} selected={hotkeysSelected} />
                  </div>
                ) : isDictionaryPage ? (
                  <div className="mt-2 space-y-1">
                    {allResults.map((result, index) => (
                      <div
                        className={`${index < 12 ? "animate-fade-up" : ""} rounded-2xl ${
                          index === hotkeysSelected ? "bg-surface ring-1 ring-accent-strong" : ""
                        }`}
                        data-hotkey-index={index}
                        key={index}
                        style={index < 12 ? { animationDelay: `${Math.min(index * 30, 300)}ms` } : undefined}
                      >
                        <DictionaryCard globals={globals} result={result} />
                      </div>
                    ))}
                  </div>
                ) : isAppsPage ? (
                  <div className="mt-4">
                    <AppsGrid globals={globals} results={allResults} selected={hotkeysSelected} />
                  </div>
                ) : isPackagesPage ? (
                  <div className="mt-4">
                    <PackageGrid globals={globals} results={allResults} selected={hotkeysSelected} />
                  </div>
                ) : isSciencePage ? (
                  <div className="mt-2 space-y-1">
                    {allResults.map((result, index) => (
                      <div
                        className={`${index < 12 ? "animate-fade-up" : ""} rounded-2xl ${
                          index === hotkeysSelected ? "bg-surface ring-1 ring-accent-strong" : ""
                        }`}
                        data-hotkey-index={index}
                        key={index}
                        style={index < 12 ? { animationDelay: `${Math.min(index * 30, 300)}ms` } : undefined}
                      >
                        <PaperCard globals={globals} result={result} />
                      </div>
                    ))}
                  </div>
                ) : isFilesPage ? (
                  <div className="mt-4">
                    <FilesGrid globals={globals} results={allResults} selected={hotkeysSelected} />
                  </div>
                ) : isProductPage ? (
                  <div className="mt-4">
                    <ProductGrid globals={globals} results={allResults} />
                  </div>
                ) : isNewsPage ? (
                  <div className="mt-2 space-y-1">
                    {allResults.map((result, index) => (
                      <div
                        className={`${index < 12 ? "animate-fade-up" : ""} rounded-2xl ${
                          index === hotkeysSelected ? "bg-surface ring-1 ring-accent-strong" : ""
                        }`}
                        data-hotkey-index={index}
                        key={index}
                        style={index < 12 ? { animationDelay: `${Math.min(index * 30, 300)}ms` } : undefined}
                      >
                        <NewsCard globals={globals} result={result} />
                      </div>
                    ))}
                  </div>
                ) : singleCategory !== null ? (
                  // category intent page: a pure relevance-ordered list in
                  // which every type keeps its own card - extracting a type
                  // into a strip would break the relevance order.
                  // space-y keeps highlighted (selected / hovered) cards from
                  // touching, matching the mixed-block and news lists.
                  <div className="mt-2 space-y-1">
                    {allResults.map((result, index) => (
                      <div
                        className={`${index < 12 ? "animate-fade-up" : ""} rounded-2xl ${
                          index === hotkeysSelected ? "bg-surface ring-1 ring-accent-strong" : ""
                        }`}
                        data-hotkey-index={index}
                        key={index}
                        style={index < 12 ? { animationDelay: `${Math.min(index * 30, 300)}ms` } : undefined}
                      >
                        <ResultCard autoOpenMap={isMapPage} eager={index < 4} globals={globals} result={result} />
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="relative mt-2">
                    {(() => {
                      // Mixed search: one collapsible block per original
                      // search category (pure relevance order inside), in tab
                      // order by default; every block renders the same full
                      // presentation as its single-category page and can be
                      // folded away via its header.
                      const blocks = collectBlocks(allResults);
                      // persisted user order first (tab order is the
                      // fallback), then categories never seen before
                      const orderedKeys = [...blocks.keys()];
                      const isCollapsed = (key: string) => Boolean(collapsedBlocks[key]);
                      const toggle = (key: string) => setCollapsedBlocks((prev) => ({ ...prev, [key]: !prev[key] }));
                      return (
                        <>
                          {orderedKeys.map((key) => {
                            const items = blocks.get(key) ?? [];
                            const collapsed = isCollapsed(key);
                            const results = items.map(({ result }) => result);
                            const indexOffset = items[0]?.index ?? 0;
                            return (
                              <section className="mt-6 first:mt-0" data-block-key={key} key={key}>
                                <GroupHeader
                                  category={key}
                                  collapsed={collapsed}
                                  count={items.length}
                                  label={globals.category_labels[key] ?? key}
                                  onToggle={() => {
                                    toggle(key);
                                  }}
                                />
                                {!collapsed ? (
                                  <div className="mt-1 space-y-1">
                                    {key === "images" ? (
                                      <ImageGrid results={results} />
                                    ) : key === "videos" ? (
                                      <VideoGrid
                                        globals={globals}
                                        indexOffset={indexOffset}
                                        results={results}
                                        selected={hotkeysSelected}
                                      />
                                    ) : key === "music" ? (
                                      <MusicGrid
                                        globals={globals}
                                        indexOffset={indexOffset}
                                        results={results}
                                        selected={hotkeysSelected}
                                      />
                                    ) : key === "files" ? (
                                      <FilesGrid
                                        globals={globals}
                                        indexOffset={indexOffset}
                                        results={results}
                                        selected={hotkeysSelected}
                                      />
                                    ) : key === "movies" ? (
                                      <PosterGrid
                                        globals={globals}
                                        indexOffset={indexOffset}
                                        results={results}
                                        selected={hotkeysSelected}
                                      />
                                    ) : key === "packages" ? (
                                      <PackageGrid
                                        globals={globals}
                                        indexOffset={indexOffset}
                                        results={results}
                                        selected={hotkeysSelected}
                                      />
                                    ) : (
                                      <div>
                                        {items.map(({ result, index }) => (
                                          <div
                                            className={`rounded-2xl ${
                                              index === hotkeysSelected ? "bg-surface ring-1 ring-accent-strong" : ""
                                            } ${index < 12 ? "animate-fade-up" : ""}`}
                                            data-hotkey-index={index}
                                            key={index}
                                            style={
                                              index < 12
                                                ? { animationDelay: `${Math.min(index * 30, 300)}ms` }
                                                : undefined
                                            }
                                          >
                                            <ResultCard
                                              autoOpenMap={isMapPage}
                                              eager={index < 4}
                                              globals={globals}
                                              result={result}
                                            />
                                          </div>
                                        ))}
                                      </div>
                                    )}
                                  </div>
                                ) : null}
                              </section>
                            );
                          })}
                        </>
                      );
                    })()}
                  </div>
                )}

                {infiniteScroll &&
                !collapsedBlocks.general &&
                appendState !== "done" &&
                (data.paging || appended.length > 0) ? (
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

          <div className="hidden w-full shrink-0 pt-4 lg:flex lg:flex-col lg:gap-3 lg:w-80 lg:pb-6">
            {showSkeletons ? null : <Sidebar data={data} onSearch={submitQuery} />}
          </div>
        </div>
      </main>

      <BackToTop />
      {helpOpen ? <HelpModal layout={settings.hotkeys} onClose={() => setHelpOpen(false)} /> : null}
    </Shell>
  );
}
