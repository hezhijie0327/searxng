// SPDX-License-Identifier: AGPL-3.0-or-later

import { useEffect, useMemo, useRef, useState } from "react";
import { HelpModal } from "../components/HelpModal.tsx";
import { ArrowUpIcon, CategoryIcon, ChevronLeftIcon, ChevronRightIcon, InfoIcon, SearchIcon } from "../components/icons.tsx";
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

/** Sections that get a Kagi/Google-style header.  Special-typed results
    interleave heavily in mixed searches (one group per engine), so each
    section type is consolidated into a single group at its first position. */
const SECTION_TEMPLATES = new Set(["images", "videos", "news"]);

/** items shown in a collapsed strip; the chevron expands to the full set */
const SECTION_CAPS: Record<string, number> = { images: 8, videos: 6, news: 6 };

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

/** Kagi/Google-style section header for a same-type result group: category
    icon + translated label + count, and a chevron that expands the group to
    its full-page layout client-side (no re-fetch). */
function GroupHeader({
  category,
  label,
  count,
  expanded,
  onToggle,
}: {
  category: string;
  label: string;
  count: number;
  expanded: boolean;
  onToggle: () => void;
}) {
  return (
    <div className="flex items-center justify-between gap-2 pb-1 pt-5 first:pt-1">
      <h2 className="flex items-center gap-1.5 text-sm font-semibold text-ink">
        <CategoryIcon category={category} className="size-4 text-accent" />
        {label}
        <span className="font-normal text-ink-3">{count}</span>
      </h2>
      <button
        aria-label={label}
        className="grid size-7 place-items-center rounded-full text-ink-3 transition-colors hover:bg-surface-2 hover:text-ink"
        onClick={onToggle}
        title={label}
        type="button"
      >
        {expanded ? <ChevronLeftIcon className="size-4" /> : <ChevronRightIcon className="size-4" />}
      </button>
    </div>
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
  const [hotkeysSelected, setHotkeysSelected] = useState(-1);
  const [expandedSection, setExpandedSection] = useState<string | null>(null);
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
                {t("meta_found")} {allResults.length} {t("meta_results")} · {t("meta_in")}{" "}
                {Math.round((data.max_response_time ?? 0) * 10) / 10} {t("seconds")}
              </p>
            ) : null}
            {!showSkeletons && data.suggestions.length > 0 ? (
              <div className="mt-2.5 flex flex-wrap items-center gap-1.5">
                {data.suggestions.slice(0, 8).map((suggestion) => (
                  <button
                    className="inline-flex items-center gap-1.5 rounded-full bg-surface-2 px-3 py-1.5 text-[13px] text-ink-2 transition-colors hover:bg-accent-soft hover:text-accent"
                    dir="auto"
                    key={suggestion.q}
                    onClick={() => {
                      submitQuery(suggestion.q);
                    }}
                    type="button"
                  >
                    <SearchIcon className="size-3.5 shrink-0 text-ink-3" />
                    <span className="truncate">{suggestion.title}</span>
                  </button>
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
                        data-hotkey-index={index}
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
                        data-hotkey-index={index}
                        key={index}
                        style={{ animationDelay: `${Math.min(index * 30, 300)}ms` }}
                      >
                        <NewsCard globals={globals} result={result} />
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="mt-2">
                    {consolidateGroups(groupResults(allResults)).map((group, groupIndex) => {
                      const label = globals.category_labels[group.template] ?? group.template;
                      if (SECTION_TEMPLATES.has(group.template)) {
                        // section chevron expands/collapses client-side from the
                        // already-fetched results (a real category search stays
                        // available via the category tabs)
                        const expanded = expandedSection === group.template;
                        const shown = expanded ? group.items : group.items.slice(0, SECTION_CAPS[group.template]);
                        return (
                          <section key={`${group.template}-${groupIndex}`}>
                            <GroupHeader
                              category={group.template}
                              count={group.items.length}
                              expanded={expanded}
                              label={label}
                              onToggle={() => {
                                setExpandedSection(expanded ? null : group.template);
                              }}
                            />
                            <div className="mt-1">
                              {group.template === "images" ? (
                                <ImageGrid results={shown.map(({ result }) => result)} />
                              ) : group.template === "videos" ? (
                                <VideoGrid globals={globals} results={shown.map(({ result }) => result)} />
                              ) : (
                                <div className="space-y-1">
                                  {shown.map(({ result, index }) => (
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
                              )}
                            </div>
                          </section>
                        );
                      }
                      if (expandedSection) {
                        // an expanded section takes over the page - hide the rest
                        return null;
                      }
                      return (
                        <div key={groupIndex}>
                          {group.items.map(({ result, index }) => (
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
                      );
                    })}
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

          {!isImagePage && !isVideoPage && !isProductPage && (data.infoboxes.length > 0 || globals.method === "POST") ? (
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
