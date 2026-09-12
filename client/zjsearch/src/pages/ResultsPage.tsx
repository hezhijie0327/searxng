// SPDX-License-Identifier: AGPL-3.0-or-later

import { type PointerEvent as ReactPointerEvent, useEffect, useMemo, useRef, useState } from "react";
import { HelpModal } from "../components/HelpModal.tsx";
import { ArrowUpIcon, CategoryIcon, ChevronDownIcon, GripVerticalIcon, InfoIcon } from "../components/icons.tsx";
import { Answers } from "../components/results/Answers.tsx";
import {
  AppsGrid,
  DictionaryCard,
  NewsCard,
  PaperCard,
  PosterGrid,
  ProductGrid,
  ResultCard,
  ResultSkeleton,
  VideoGrid,
} from "../components/results/cards.tsx";
import { FilesGrid } from "../components/results/FilesGrid.tsx";
import { ImageGrid } from "../components/results/ImageGrid.tsx";
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

function NoResults({ pageno, hasInfobox }: { pageno: number; hasInfobox: boolean }) {
  const t = useT();
  const firstPage = pageno === 1;
  if (hasInfobox && firstPage) {
    return (
      <div className="rounded-2xl border border-line bg-surface p-4 text-sm text-ink-2">
        <p className="flex items-center gap-2">
          <InfoIcon className="size-4 shrink-0 text-accent" />
          {t("no_web_results")}
        </p>
      </div>
    );
  }
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

/** Packages fold into the it block; every other category stands alone. */
function blockKeyOf(result: ResultItem): string {
  const category = result.category || "general";
  return category === "packages" ? "it" : category;
}

function collectBlocks(results: ResultItem[]): Map<string, Array<{ result: ResultItem; index: number }>> {
  const blocks = new Map<string, Array<{ result: ResultItem; index: number }>>();
  results.forEach((result, index) => {
    const key = blockKeyOf(result);
    const items = blocks.get(key);
    if (items) {
      items.push({ result, index });
    } else {
      blocks.set(key, [{ result, index }]);
    }
  });
  return blocks;
}

/** Collapsible block header: category icon + translated label + result
    count; the whole header toggles the block. */
interface GripHandlers {
  onPointerDown: (event: ReactPointerEvent<HTMLSpanElement>) => void;
  onPointerUp: (event: ReactPointerEvent<HTMLSpanElement>) => void;
  onKeyDown: (event: React.KeyboardEvent<HTMLSpanElement>) => void;
}

function GroupHeader({
  category,
  label,
  count,
  collapsed,
  onToggle,
  grip,
}: {
  category: string;
  label: string;
  count: number;
  collapsed: boolean;
  onToggle: () => void;
  grip?: GripHandlers;
}) {
  const t = useT();
  return (
    <h2 className="group flex items-center gap-1 pb-1 pt-2">
      {grip ? (
        <span
          aria-label={t("drag_reorder")}
          className="-ms-1 cursor-grab touch-none rounded p-1 text-ink-3 transition-colors hover:bg-surface-2 hover:text-ink active:cursor-grabbing"
          onKeyDown={grip.onKeyDown}
          onPointerDown={grip.onPointerDown}
          onPointerUp={grip.onPointerUp}
          role="button"
          tabIndex={0}
          title={t("drag_reorder")}
        >
          <GripVerticalIcon className="size-4" />
        </span>
      ) : null}
      <button
        aria-expanded={!collapsed}
        className="flex min-w-0 flex-1 items-center gap-1.5 text-left text-sm font-semibold text-ink"
        onClick={onToggle}
        type="button"
      >
        <CategoryIcon category={category} className="size-4 shrink-0 text-accent" />
        {label}
        <span className="font-normal text-ink-3">{count}</span>
        <ChevronDownIcon
          className={`size-4 shrink-0 text-ink-3 transition-transform ${collapsed ? "-rotate-90" : ""}`}
        />
      </button>
    </h2>
  );
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
              <p className="mt-2 ps-3.5 text-xs text-ink-3">
                {t("meta_found")} {allResults.length} {t("meta_results")}
              </p>
            ) : null}
            {!showSkeletons ? (
              <div className="mt-2 flex flex-col gap-3 lg:hidden">
                <SuggestionsBox data={data} onSearch={submitQuery} />
                <DebugPanels data={data} />
              </div>
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
                  <Answers answers={calcAnswer ? [calcAnswer, ...data.answers] : data.answers} query={data.q} />
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
            {showSkeletons ? null : <SuggestionsBox data={data} onSearch={submitQuery} />}
            {showSkeletons ? null : <DebugPanels data={data} />}
          </div>
        </div>
      </main>

      <BackToTop />
      {helpOpen ? <HelpModal layout={settings.hotkeys} onClose={() => setHelpOpen(false)} /> : null}
    </Shell>
  );
}
