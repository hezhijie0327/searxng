// SPDX-License-Identifier: AGPL-3.0-or-later

import { useEffect, useMemo, useRef, useState } from "react";
import { HelpModal } from "../components/HelpModal.tsx";
import { ArrowUpIcon, CategoryIcon, ChevronDownIcon, ChevronUpIcon, InfoIcon } from "../components/icons.tsx";
import { Answers } from "../components/results/Answers.tsx";
import {
  NewsCard,
  PaperCard,
  ProductGrid,
  ResultCard,
  ResultSkeleton,
  VideoGrid,
} from "../components/results/cards.tsx";
import { FilesGrid } from "../components/results/FilesGrid.tsx";
import { ImageGrid, ImageStrip } from "../components/results/ImageGrid.tsx";
import { MusicGrid } from "../components/results/MusicGrid.tsx";
import { PackageGrid } from "../components/results/PackageGrid.tsx";
import { Pagination } from "../components/results/Pagination.tsx";
import { DebugPanels, Infobox, Sidebar, SuggestionsBox } from "../components/results/Sidebar.tsx";
import { Strip } from "../components/results/Strip.tsx";
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

/** Mixed searches group results by template; music results that carry the
    default template (genius, ...) and torrent/file downloads belong to
    their own sections so every category shares one presentation. */
function groupKey(result: ResultItem): string {
  const template = result.template || "default";
  if (template === "default" && result.category === "music") {
    return "music";
  }
  if (template === "torrent" || template === "file") {
    return "files";
  }
  return template;
}

/** Consecutive same-template results form groups (image strips in mixed mode). */
function groupResults(
  results: ResultItem[],
): Array<{ template: string; items: Array<{ result: ResultItem; index: number }> }> {
  const groups: Array<{ template: string; items: Array<{ result: ResultItem; index: number }> }> = [];
  for (let index = 0; index < results.length; index += 1) {
    const result = results[index];
    const template = result ? groupKey(result) : "default";
    const last = groups[groups.length - 1];
    if (last && last.template === template) {
      last.items.push({ result: result as ResultItem, index });
    } else {
      groups.push({ template, items: [{ result: result as ResultItem, index }] });
    }
  }
  return groups;
}

/** Sections that get a Kagi/Google-style header.  They render as fixed-row
    horizontal strips (paged with left/right arrows) in this fixed order
    after the untyped results - nothing expands in place. */
const SECTION_TEMPLATES = new Set(["general", "images", "videos", "news", "music", "files", "packages"]);
const SECTION_ORDER = ["images", "videos", "news", "music", "files", "packages"];
const BLOCK_ORDER_KEY = "zjs-block-order";

function consolidateGroups(
  groups: Array<{ template: string; items: Array<{ result: ResultItem; index: number }> }>,
): Array<{ template: string; items: Array<{ result: ResultItem; index: number }> }> {
  const first: Map<string, { template: string; items: Array<{ result: ResultItem; index: number }> }> = new Map();
  const out: Array<{ template: string; items: Array<{ result: ResultItem; index: number }> }> = [];
  for (const group of groups) {
    if (!SECTION_TEMPLATES.has(group.template)) {
      out.push(group);
      continue;
    }
    const merged = first.get(group.template);
    if (merged) {
      merged.items.push(...group.items);
    } else {
      const created = { template: group.template, items: [...group.items] };
      first.set(group.template, created);
      out.push(created);
    }
  }
  return out;
}

/** Collapsible block header: category icon + translated label + result
    count; the whole header toggles the block. */
function GroupHeader({
  category,
  label,
  count,
  collapsed,
  onToggle,
  onMoveUp,
  onMoveDown,
}: {
  category: string;
  label: string;
  count: number;
  collapsed: boolean;
  onToggle: () => void;
  onMoveUp?: (() => void) | undefined;
  onMoveDown?: (() => void) | undefined;
}) {
  const t = useT();
  return (
    <h2 className="group flex items-center gap-1 pb-1 pt-2">
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
      {onMoveUp ? (
        <button
          aria-label={`${t("move_up")} ${label}`}
          className="grid size-6 shrink-0 place-items-center rounded-full text-ink-3 opacity-40 transition-all hover:bg-surface-2 hover:text-ink hover:opacity-100 focus-visible:opacity-100"
          onClick={onMoveUp}
          title={t("move_up")}
          type="button"
        >
          <ChevronUpIcon className="size-3.5" />
        </button>
      ) : null}
      {onMoveDown ? (
        <button
          aria-label={`${t("move_down")} ${label}`}
          className="grid size-6 shrink-0 place-items-center rounded-full text-ink-3 opacity-40 transition-all hover:bg-surface-2 hover:text-ink hover:opacity-100 focus-visible:opacity-100"
          onClick={onMoveDown}
          title={t("move_down")}
          type="button"
        >
          <ChevronDownIcon className="size-3.5" />
        </button>
      ) : null}
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
  const [blockOrder, setBlockOrder] = useState<string[]>(() => {
    try {
      const stored: unknown = JSON.parse(localStorage.getItem(BLOCK_ORDER_KEY) ?? "null");
      if (Array.isArray(stored) && stored.every((key) => typeof key === "string")) {
        return stored;
      }
    } catch {}
    return ["general", ...SECTION_ORDER];
  });
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
  const isImagePage =
    (data.only_template === "images" || singleCategory === "images") &&
    allResults.every((result) => result.template === "images" || result.thumbnail_src || result.img_src);
  const isVideoPage = data.only_template === "videos" || singleCategory === "videos";
  const isProductPage = (data.only_template === "products" || singleCategory === "products") && !isVideoPage;
  const isNewsPage = singleCategory === "news" && !isImagePage && !isVideoPage;
  const isMapPage = singleCategory === "map" && !isImagePage && !isVideoPage;
  const isMusicPage = singleCategory === "music" && !isImagePage && !isVideoPage;
  // science intent renders every result in the scholarly layout; a
  // paper-only bang search (`!pubmed ...`) gets the same treatment
  const isSciencePage =
    (singleCategory === "science" || data.only_template === "paper") && !isImagePage && !isVideoPage && !isMusicPage;
  // files intent (or a torrent-only bang search) gets the file-tile grid;
  // torrents keep the transfer card inside mixed searches
  const isFilesPage =
    (singleCategory === "files" || data.only_template === "torrent") &&
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
                  <Answers answers={calcAnswer ? [calcAnswer, ...data.answers] : data.answers} />
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
                ) : isSciencePage ? (
                  <div className="mt-2 space-y-1">
                    {allResults.map((result, index) => (
                      <div
                        className={`animate-fade-up rounded-2xl ${
                          index === hotkeysSelected ? "bg-surface ring-1 ring-accent-strong" : ""
                        }`}
                        data-hotkey-index={index}
                        key={index}
                        style={{ animationDelay: `${Math.min(index * 30, 300)}ms` }}
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
                        className={`animate-fade-up rounded-2xl ${
                          index === hotkeysSelected ? "bg-surface ring-1 ring-accent-strong" : ""
                        }`}
                        data-hotkey-index={index}
                        key={index}
                        style={{ animationDelay: `${Math.min(index * 30, 300)}ms` }}
                      >
                        <NewsCard globals={globals} result={result} />
                      </div>
                    ))}
                  </div>
                ) : singleCategory !== null ? (
                  // category intent page: a pure relevance-ordered list in
                  // which every type keeps its own card - extracting a type
                  // into a strip would break the relevance order
                  <div className="mt-2">
                    {allResults.map((result, index) => (
                      <div
                        className={`animate-fade-up rounded-2xl ${
                          index === hotkeysSelected ? "bg-surface ring-1 ring-accent-strong" : ""
                        }`}
                        data-hotkey-index={index}
                        key={index}
                        style={{ animationDelay: `${Math.min(index * 30, 300)}ms` }}
                      >
                        <ResultCard autoOpenMap={isMapPage} eager={index < 4} globals={globals} result={result} />
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="mt-2">
                    {(() => {
                      // Mixed search: every type renders as its own
                      // collapsible block - the untyped web list first (pure
                      // relevance order), then the typed strips.  The block
                      // order is user-adjustable (headers expose up/down) and
                      // persists in localStorage; collapsing works on every
                      // screen size so huge blocks never dominate the page.
                      const groups = consolidateGroups(groupResults(allResults));
                      const rest = groups.filter((group) => !SECTION_TEMPLATES.has(group.template));
                      const restCount = rest.reduce((sum, group) => sum + group.items.length, 0);
                      const sectionByKey = new Map(
                        groups
                          .filter((group) => SECTION_TEMPLATES.has(group.template))
                          .map((group) => [group.template, group]),
                      );
                      const defaultOrder = ["general", ...SECTION_ORDER];
                      const orderedKeys = [
                        ...blockOrder.filter(
                          (key) => defaultOrder.includes(key) && (key === "general" || sectionByKey.has(key)),
                        ),
                        ...defaultOrder.filter(
                          (key) => !blockOrder.includes(key) && (key === "general" || sectionByKey.has(key)),
                        ),
                      ];
                      const isCollapsed = (key: string) => Boolean(collapsedBlocks[key]);
                      const toggle = (key: string) => setCollapsedBlocks((prev) => ({ ...prev, [key]: !prev[key] }));
                      const move = (key: string, delta: -1 | 1) => {
                        const order = [...blockOrder];
                        const from = order.indexOf(key);
                        const to = from + delta;
                        if (from < 0 || to < 0 || to >= order.length) {
                          return;
                        }
                        const moved = order.splice(from, 1)[0];
                        if (moved !== undefined) {
                          order.splice(to, 0, moved);
                        }
                        setBlockOrder(order);
                        try {
                          localStorage.setItem(BLOCK_ORDER_KEY, JSON.stringify(order));
                        } catch {}
                      };
                      const restItems = rest.flatMap((group) => group.items);
                      const restBlock = {
                        key: "general",
                        node: (
                          <section className="mt-6 first:mt-0" key="general">
                            <GroupHeader
                              category="general"
                              collapsed={Boolean(collapsedBlocks.general)}
                              count={restCount}
                              label={globals.category_labels.general ?? "general"}
                              onMoveDown={
                                orderedKeys.indexOf("general") < orderedKeys.length - 1
                                  ? () => {
                                      move("general", 1);
                                    }
                                  : undefined
                              }
                              onToggle={() => {
                                toggle("general");
                              }}
                            />
                            {!collapsedBlocks.general ? (
                              <div className="mt-1">
                                {restItems.map(({ result, index }) => (
                                  <div
                                    className={`animate-fade-up rounded-2xl ${
                                      index === hotkeysSelected ? "bg-surface ring-1 ring-accent-strong" : ""
                                    }`}
                                    data-hotkey-index={index}
                                    key={index}
                                    style={{ animationDelay: `${Math.min(index * 30, 300)}ms` }}
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
                            ) : null}
                          </section>
                        ),
                      };
                      return (
                        <>
                          {orderedKeys.map((key, position) => {
                            if (key === "general") {
                              return restBlock.node;
                            }
                            const group = sectionByKey.get(key);
                            if (!group) {
                              return null;
                            }
                            const label = globals.category_labels[key] ?? t(key);
                            const results = group.items.map(({ result }) => result);
                            const indexOffset = group.items[0]?.index ?? 0;
                            const collapsed = isCollapsed(key);
                            return (
                              <section className="mt-6 first:mt-0" key={key}>
                                <GroupHeader
                                  category={key}
                                  collapsed={collapsed}
                                  count={group.items.length}
                                  label={label}
                                  onMoveDown={
                                    position < orderedKeys.length - 1
                                      ? () => {
                                          move(key, 1);
                                        }
                                      : undefined
                                  }
                                  onMoveUp={
                                    position > 0
                                      ? () => {
                                          move(key, -1);
                                        }
                                      : undefined
                                  }
                                  onToggle={() => {
                                    toggle(key);
                                  }}
                                />
                                {!collapsed ? (
                                  <div className="mt-1">
                                    {key === "images" ? (
                                      <ImageStrip results={results} />
                                    ) : key === "videos" ? (
                                      <VideoGrid
                                        globals={globals}
                                        indexOffset={indexOffset}
                                        results={results}
                                        selected={hotkeysSelected}
                                        variant="strip"
                                      />
                                    ) : key === "music" ? (
                                      <MusicGrid
                                        globals={globals}
                                        indexOffset={indexOffset}
                                        results={results}
                                        selected={hotkeysSelected}
                                        variant="strip"
                                      />
                                    ) : key === "files" ? (
                                      <FilesGrid
                                        globals={globals}
                                        indexOffset={indexOffset}
                                        results={results}
                                        selected={hotkeysSelected}
                                        variant="strip"
                                      />
                                    ) : key === "packages" ? (
                                      <PackageGrid
                                        globals={globals}
                                        indexOffset={indexOffset}
                                        results={results}
                                        selected={hotkeysSelected}
                                        variant="strip"
                                      />
                                    ) : (
                                      <Strip rows={1}>
                                        {group.items.map(({ result, index }) => (
                                          <div
                                            className={`h-full rounded-2xl ${
                                              index === hotkeysSelected ? "bg-surface ring-1 ring-accent-strong" : ""
                                            }`}
                                            data-hotkey-index={index}
                                            key={index}
                                          >
                                            <NewsCard globals={globals} result={result} />
                                          </div>
                                        ))}
                                      </Strip>
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
