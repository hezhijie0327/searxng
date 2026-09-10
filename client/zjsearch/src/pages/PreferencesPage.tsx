// SPDX-License-Identifier: AGPL-3.0-or-later

import { type ReactNode, useEffect, useMemo, useRef, useState } from "react";
import type { DropdownOption } from "../components/Dropdown.tsx";
import { Dropdown } from "../components/Dropdown.tsx";
import {
  AlertIcon,
  BookIcon,
  CategoryIcon,
  CenterIcon,
  CheckIcon,
  CloseIcon,
  CookieIcon,
  ExternalLinkIcon,
  GlobeIcon,
  GridIcon,
  ImageIcon,
  KeyboardIcon,
  KeyIcon,
  LanguagesIcon,
  LinkIcon,
  MoonIcon,
  RefreshIcon,
  SearchIcon,
  ShieldIcon,
  SlidersIcon,
  SparkIcon,
  StarIcon,
  SunIcon,
  SwapIcon,
  TerminalIcon,
} from "../components/icons.tsx";
import { Link, Shell } from "../components/Shell.tsx";
import { loadEngineDescriptions } from "../lib/engineDescriptions.ts";
import { useT } from "../lib/i18n.ts";
import type { ThemeStyle } from "../lib/theme.ts";
import { applyCenterAlignment, applyThemeStyle } from "../lib/theme.ts";
import type { EngineEntry, PreferencesPageData } from "../lib/types.ts";

// ------------------------------------------------------------ layout blocks

function cap(value: string): string {
  return value.charAt(0).toUpperCase() + value.slice(1);
}

function IconTile({ children }: { children: ReactNode }) {
  return (
    <span className="grid size-10 shrink-0 place-items-center rounded-xl bg-accent-soft text-accent">{children}</span>
  );
}

/** One settings row: icon tile + title/description on the left, control on the right. */
function SettingRow({
  icon,
  title,
  description,
  children,
  stacked,
}: {
  icon: ReactNode;
  title: string;
  description?: string;
  children: ReactNode;
  stacked?: boolean;
}) {
  if (stacked) {
    return (
      <div className="px-5 py-5 transition-colors hover:bg-surface-2/40 sm:px-6">
        <div className="flex items-center gap-4">
          <IconTile>{icon}</IconTile>
          <div className="min-w-0">
            <p className="text-sm font-medium text-ink">{title}</p>
            {description ? <p className="mt-0.5 text-xs leading-relaxed text-ink-3">{description}</p> : null}
          </div>
        </div>
        <div className="mt-4 sm:pl-14">{children}</div>
      </div>
    );
  }
  return (
    <div className="flex flex-col gap-3 px-5 py-5 transition-colors hover:bg-surface-2/40 sm:flex-row sm:items-center sm:justify-between sm:gap-8 sm:px-6">
      <div className="flex min-w-0 items-center gap-4">
        <IconTile>{icon}</IconTile>
        <div className="min-w-0">
          <p className="text-sm font-medium text-ink">{title}</p>
          {description ? <p className="mt-0.5 text-xs leading-relaxed text-ink-3">{description}</p> : null}
        </div>
      </div>
      <div className="shrink-0">{children}</div>
    </div>
  );
}

function Card({ children }: { children: ReactNode }) {
  return (
    <div className="divide-y divide-line overflow-hidden rounded-2xl border border-line bg-surface animate-fade-up">
      {children}
    </div>
  );
}

function Switch({
  checked,
  onChange,
  label,
}: {
  checked: boolean;
  onChange: (checked: boolean) => void;
  label: string;
}) {
  return (
    <button
      aria-checked={checked}
      aria-label={label}
      className={`relative inline-flex h-6 w-11 shrink-0 cursor-pointer items-center rounded-full p-0.5 transition-colors focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent ${
        checked ? "bg-accent-strong" : "bg-surface-2 ring-1 ring-line"
      }`}
      onClick={() => {
        onChange(!checked);
      }}
      role="switch"
      type="button"
    >
      <span
        className={`size-5 rounded-full shadow transition-transform ${
          checked ? "translate-x-5 bg-accent-contrast" : "translate-x-0 bg-ink-3"
        }`}
      />
    </button>
  );
}

function Select({
  value,
  options,
  onChange,
  ariaLabel,
}: {
  value: string;
  options: DropdownOption[];
  onChange: (value: string) => void;
  ariaLabel?: string;
}) {
  return (
    <div className="w-full sm:w-60">
      <Dropdown align="end" ariaLabel={ariaLabel} onChange={onChange} options={options} value={value} variant="boxed" />
    </div>
  );
}

function PluginRow({
  plugin,
  enabled,
  onChange,
}: {
  plugin: { id: string; name: string; description: string };
  enabled: boolean;
  onChange: (checked: boolean) => void;
}) {
  return (
    <SettingRow description={plugin.description} icon={<TerminalIcon className="size-4.5" />} title={plugin.name}>
      <Switch checked={enabled} label={plugin.name} onChange={onChange} />
    </SettingRow>
  );
}

// ------------------------------------------------------------------ engines

function reliabilityColor(reliability: number | null): string {
  if (reliability === null) {
    return "text-ink-3";
  }
  if (reliability <= 50) {
    return "text-danger";
  }
  if (reliability < 80) {
    return "text-warning";
  }
  if (reliability < 90) {
    return "text-ink-2";
  }
  return "text-ok";
}

function EngineTooltip({ engine }: { engine: EngineEntry }) {
  const [desc, setDesc] = useState<{ text: string; source: string } | null>(null);
  useEffect(() => {
    void loadEngineDescriptions().then((map) => {
      const entry = map[engine.name];
      if (entry) {
        setDesc({ text: entry[0], source: entry[1] });
      }
    });
  }, [engine.name]);

  return (
    <div className="pointer-events-none absolute start-0 top-full z-30 mt-1 hidden w-80 rounded-xl border border-line bg-surface p-3 text-xs shadow-pop group-hover/engine:block">
      {desc ? (
        <p className="text-ink-2">
          {desc.text} <i className="text-ink-3">(Source: {desc.source})</i>
        </p>
      ) : (
        <p className="text-ink-3">…</p>
      )}
      {engine.website ? (
        <p className="mt-1.5 truncate">
          <a
            className="inline-flex items-center gap-1 text-accent hover:underline"
            href={engine.website}
            rel="noreferrer"
            target="_blank"
          >
            {engine.website}
            <ExternalLinkIcon className="size-3" />
          </a>
        </p>
      ) : null}
      {engine.enable_http ? (
        <p className="mt-1.5 inline-flex items-center gap-1 text-warning">
          <AlertIcon className="size-3.5" /> No HTTPS
        </p>
      ) : null}
      <p className="mt-1.5 flex flex-wrap gap-1">
        <span className="text-ink-3">!bang:</span>
        {[engine.name, engine.shortcut].map((bang) => (
          <code className="rounded bg-surface-2 px-1" key={bang}>
            !{bang.replaceAll(" ", "_")}
          </code>
        ))}
      </p>
      {engine.errors.length > 0 ? (
        <p className="mt-1.5">
          <Link className="text-accent hover:underline" href={`/stats?engine=${encodeURIComponent(engine.name)}`}>
            View error logs and submit a bug report
          </Link>
        </p>
      ) : null}
    </div>
  );
}

function EnginesTab({
  tab,
  enabled,
  toggleEngine,
  setAll,
  showMetrics,
}: {
  tab: PreferencesPageData["engine_tabs"][number];
  enabled: Record<string, boolean>;
  toggleEngine: (key: string, value: boolean) => void;
  setAll: (keys: string[], value: boolean) => void;
  showMetrics: boolean;
}) {
  const t = useT();
  const keys = tab.groups.flatMap((group) => group.engines.map((engine) => `${engine.name}__${tab.category}`));
  return (
    <div>
      <div className="mb-3 flex items-center gap-2">
        <button
          className="inline-flex items-center gap-1.5 rounded-full bg-surface-2 px-3 py-1.5 text-xs text-ink-2 transition-colors hover:text-ink"
          onClick={() => {
            setAll(keys, true);
          }}
          type="button"
        >
          <CheckIcon className="size-3.5" />
          {t("enable_all")}
        </button>
        <button
          className="inline-flex items-center gap-1.5 rounded-full bg-surface-2 px-3 py-1.5 text-xs text-ink-2 transition-colors hover:text-ink"
          onClick={() => {
            setAll(keys, false);
          }}
          type="button"
        >
          <CloseIcon className="size-3.5" />
          {t("disable_all")}
        </button>
      </div>
      <div className="overflow-x-auto rounded-2xl border border-line">
        <table className="w-full min-w-[680px] text-left text-xs">
          <thead className="bg-surface-2 text-ink-3">
            <tr>
              <th className="px-4 py-3 font-medium">{t("allow")}</th>
              <th className="px-4 py-3 font-medium">{t("engine_name")}</th>
              <th className="px-4 py-3 font-medium">{t("bang")}</th>
              <th className="px-4 py-3 font-medium">{t("safesearch")}</th>
              <th className="px-4 py-3 font-medium">{t("time_range")}</th>
              <th className="px-4 py-3 font-medium">{t("weight")}</th>
              {showMetrics ? <th className="px-4 py-3 font-medium">{t("response_time")}</th> : null}
              <th className="px-4 py-3 font-medium">{t("max_time")}</th>
              {showMetrics ? <th className="px-4 py-3 font-medium">{t("reliability")}</th> : null}
            </tr>
          </thead>
          <tbody>
            {tab.groups.flatMap((group) => {
              const rows: ReactNode[] = [];
              if (group.engines.length > 1) {
                rows.push(
                  <tr className="bg-surface-2/60" key={`group-${group.group}`}>
                    <td className="px-3 py-1.5 font-medium text-ink-2" colSpan={2}>
                      {group.group}
                    </td>
                    <td className="px-3 py-1.5" colSpan={showMetrics ? 7 : 5}>
                      {group.group_bang ? <code className="rounded bg-surface-2 px-1">{group.group_bang}</code> : null}
                    </td>
                  </tr>,
                );
              }
              for (const engine of group.engines) {
                const key = `${engine.name}__${tab.category}`;
                rows.push(
                  <tr className="border-t border-line transition-colors hover:bg-surface-2/40" key={key}>
                    <td className="px-3 py-3">
                      <Switch
                        checked={enabled[key] ?? false}
                        label={`Allow ${engine.name}`}
                        onChange={(value) => {
                          toggleEngine(key, value);
                        }}
                      />
                    </td>
                    <td className="max-w-52 px-4 py-3">
                      <div className="group/engine relative">
                        <button
                          className="flex items-center gap-1 truncate font-medium text-ink"
                          onMouseEnter={() => void loadEngineDescriptions()}
                          type="button"
                        >
                          {engine.enable_http ? <AlertIcon className="size-3.5 shrink-0 text-warning" /> : null}
                          <span className="truncate">
                            {engine.name}
                            {engine.language ? ` (${engine.language})` : ""}
                          </span>
                        </button>
                        <EngineTooltip engine={engine} />
                      </div>
                    </td>
                    <td className="px-4 py-3">
                      <code className="rounded bg-surface-2 px-1">!{engine.shortcut}</code>
                    </td>
                    <td className="px-4 py-3">
                      {engine.supports_safesearch ? "✓" : <span className="text-ink-3">–</span>}
                    </td>
                    <td className="px-4 py-3">
                      {engine.supports_time_range ? "✓" : <span className="text-ink-3">–</span>}
                    </td>
                    <td className="px-4 py-3">{engine.weight}</td>
                    {showMetrics ? (
                      <td className="px-4 py-3">
                        {engine.stats_time !== null ? (
                          <div className="flex items-center gap-2">
                            <span className="w-10 text-ink-2">{engine.stats_time}</span>
                            <span className="h-1.5 w-24 overflow-hidden rounded-full bg-surface-2">
                              <span
                                className="block h-full bg-accent-strong"
                                style={{ width: `${Math.min(100, engine.stats_time)}%` }}
                              />
                            </span>
                          </div>
                        ) : (
                          <span className="text-ink-3">–</span>
                        )}
                      </td>
                    ) : null}
                    <td className={`px-4 py-3 ${engine.warn_timeout ? "font-medium text-danger" : "text-ink-2"}`}>
                      {engine.timeout}s
                    </td>
                    {showMetrics ? (
                      <td className={`px-4 py-3 font-medium ${reliabilityColor(engine.reliability)}`}>
                        {engine.reliability ?? "–"}
                      </td>
                    ) : null}
                  </tr>,
                );
              }
              return rows;
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

// --------------------------------------------------------------------- page

export function PreferencesPage({ data, embedded = false }: { data: PreferencesPageData; embedded?: boolean }) {
  const t = useT();
  const globals = data.globals;
  const kv = data.kv;
  const locked = useMemo(() => new Set(data.locked_preferences), [data.locked_preferences]);

  const [tab, setTab] = useState<"general" | "ui" | "privacy" | "engines" | "query" | "cookies">("general");
  const [engineTab, setEngineTab] = useState(0);
  const [savedAt, setSavedAt] = useState(0);

  // form state
  const [categories, setCategories] = useState<string[]>(kv.categories);
  const [language, setLanguage] = useState(kv.language);
  const [autocomplete, setAutocomplete] = useState(kv.autocomplete);
  const [faviconResolver, setFaviconResolver] = useState(kv.favicon_resolver);
  const [safesearch, setSafesearch] = useState(String(kv.safesearch));
  const [tokens, setTokens] = useState(kv.tokens);
  const [doiResolver, setDoiResolver] = useState(
    Object.entries(data.doi_resolvers).find(([, url]) => url === kv.doi_resolver)?.[0] ?? "",
  );
  const [locale, setLocale] = useState(kv.locale);
  const [theme, setTheme] = useState(kv.theme);
  const [themeStyle, setThemeStyle] = useState<string>(kv.simple_style);
  const [resultsOnNewTab, setResultsOnNewTab] = useState(kv.results_on_new_tab);
  const [searchOnCategorySelect, setSearchOnCategorySelect] = useState(kv.search_on_category_select);
  const [hotkeys, setHotkeys] = useState(kv.hotkeys);
  const [urlFormatting, setUrlFormatting] = useState(kv.url_formatting);
  const [method, setMethod] = useState<"GET" | "POST">(kv.method);
  const [imageProxy, setImageProxy] = useState(kv.image_proxy);
  const [queryInTitle, setQueryInTitle] = useState(kv.query_in_title);
  const [centerAlignment, setCenterAlignment] = useState(kv.center_alignment);
  const [engines, setEngines] = useState<Record<string, boolean>>(() => {
    const map: Record<string, boolean> = {};
    for (const tabInfo of data.engine_tabs) {
      for (const group of tabInfo.groups) {
        for (const engine of group.engines) {
          map[`${engine.name}__${tabInfo.category}`] = !engine.disabled;
        }
      }
    }
    return map;
  });
  const [plugins, setPlugins] = useState<Record<string, boolean>>(() => {
    const map: Record<string, boolean> = {};
    for (const plugin of data.plugins) {
      map[plugin.id] = plugin.enabled;
    }
    return map;
  });
  const [pastedHash, setPastedHash] = useState("");

  const isPreview = new URLSearchParams(window.location.search).get("preferences_preview_only") === "true";
  const locales = Object.entries(data.locales).sort((a, b) => a[1].localeCompare(b[1]));
  const showMetrics = globals.enable_metrics;
  const currentEngineTab = data.engine_tabs[engineTab];

  const toggleEngine = (key: string, value: boolean) => {
    setEngines((prev) => ({ ...prev, [key]: value }));
  };
  const setAll = (keys: string[], value: boolean) => {
    setEngines((prev) => {
      const next = { ...prev };
      for (const key of keys) {
        next[key] = value;
      }
      return next;
    });
  };

  // snapshot of every saved field, used to detect changes
  const formSignature = JSON.stringify([
    categories,
    language,
    autocomplete,
    faviconResolver,
    safesearch,
    tokens,
    doiResolver,
    locale,
    theme,
    themeStyle,
    hotkeys,
    urlFormatting,
    method,
    imageProxy,
    queryInTitle,
    centerAlignment,
    resultsOnNewTab,
    searchOnCategorySelect,
    engines,
    plugins,
    pastedHash,
  ]);

  const initialized = useRef(false);

  // live-apply: every change is debounced and POSTed with the exact form
  // semantics of upstream /preferences: absent booleans are false, and the
  // `engine_*` / `plugin_*` keys are REVERSED — a posted key marks that engine
  // or plugin as disabled, so we send exactly the disabled set (omitted keys
  // are re-enabled by the server).
  // biome-ignore lint/correctness/useExhaustiveDependencies: formSignature covers all saved fields
  useEffect(() => {
    // skip the initial mount: nothing changed yet, posting would corrupt
    // the reversed engine/plugin sets
    if (!initialized.current) {
      initialized.current = true;
      return;
    }
    const timer = window.setTimeout(() => {
      const fd = new FormData();
      fd.set("language", language);
      fd.set("autocomplete", autocomplete);
      fd.set("favicon_resolver", faviconResolver);
      fd.set("safesearch", safesearch);
      fd.set("locale", locale);
      fd.set("theme", theme);
      fd.set("simple_style", themeStyle);
      fd.set("hotkeys", hotkeys);
      fd.set("url_formatting", urlFormatting);
      fd.set("method", method);
      fd.set("doi_resolver", doiResolver);
      fd.set("tokens", tokens);
      if (resultsOnNewTab) {
        fd.set("results_on_new_tab", "on");
      }
      if (searchOnCategorySelect) {
        fd.set("search_on_category_select", "on");
      }
      if (imageProxy) {
        fd.set("image_proxy", "on");
      }
      if (queryInTitle) {
        fd.set("query_in_title", "on");
      }
      if (centerAlignment) {
        fd.set("center_alignment", "on");
      }
      for (const category of categories) {
        fd.append(`category_${category}`, "on");
      }
      for (const [key, allowed] of Object.entries(engines)) {
        if (!allowed) {
          fd.set(`engine_${key.replaceAll(" ", "_")}`, "on");
        }
      }
      for (const [id, enabled] of Object.entries(plugins)) {
        if (!enabled) {
          fd.set(`plugin_${id}`, "on");
        }
      }
      if (pastedHash.trim()) {
        fd.set("preferences", pastedHash.trim());
      }
      void fetch("/preferences", { method: "POST", body: fd, redirect: "follow" })
        .then(() => {
          setSavedAt(Date.now());
        })
        .catch(() => {
          /* keep the UI state; changing any setting retries */
        });
    }, 600);
    return () => {
      window.clearTimeout(timer);
    };
  }, [formSignature]);

  const pluginSections = (section: string) => data.plugins.filter((plugin) => plugin.section === section);
  const shareOrigin = window.location.origin;

  const tabs = [
    { id: "general", label: t("general"), icon: <SlidersIcon className="size-4" /> },
    { id: "ui", label: t("user_interface"), icon: <SunIcon className="size-4" /> },
    { id: "privacy", label: t("privacy"), icon: <ShieldIcon className="size-4" /> },
    { id: "engines", label: t("engines"), icon: <GridIcon className="size-4" /> },
    { id: "query", label: t("special_queries"), icon: <TerminalIcon className="size-4" /> },
    { id: "cookies", label: t("cookies"), icon: <CookieIcon className="size-4" /> },
  ] as const;

  return (
    <Shell embedded={embedded} globals={globals}>
      <main className="mx-auto w-full max-w-4xl flex-1 px-4 pb-20 sm:px-6">
        <div className="flex items-center justify-between py-6">
          <h1 className="text-2xl font-semibold tracking-tight text-ink">{t("preferences")}</h1>
          <div className="flex items-center gap-3">
            {savedAt > 0 ? (
              <span className="inline-flex items-center gap-1 text-xs text-ok animate-fade-in">
                <CheckIcon className="size-3.5" />
                {t("saved")}
              </span>
            ) : null}
          </div>
        </div>

        {isPreview ? (
          <div className="mb-4 flex items-start gap-3 rounded-2xl border border-warning/40 bg-warning/10 p-4 text-sm text-ink-2 animate-fade-up">
            <AlertIcon className="mt-0.5 size-4 shrink-0 text-warning" />
            <div>
              <p>{t("preview_banner")}</p>
              <ul className="mt-1.5 list-disc ps-5">
                <li>{t("press_save_to_copy")}</li>
                <li>
                  {t("view_browser_prefs")}{" "}
                  <Link className="text-accent hover:underline" href="/preferences">
                    /preferences
                  </Link>
                </li>
              </ul>
            </div>
          </div>
        ) : null}

        <div className="mb-6 flex flex-wrap gap-1.5 rounded-2xl border border-line bg-surface p-2">
          {tabs.map((item) => (
            <button
              aria-selected={tab === item.id}
              className={`flex flex-1 items-center justify-center gap-2 whitespace-nowrap rounded-xl px-4 py-2.5 text-sm transition-colors ${
                tab === item.id
                  ? "bg-accent-strong font-medium text-accent-contrast"
                  : "text-ink-2 hover:bg-surface-2 hover:text-ink"
              }`}
              key={item.id}
              onClick={() => {
                setTab(item.id);
              }}
              role="tab"
              type="button"
            >
              {item.icon}
              <span>{item.label}</span>
            </button>
          ))}
        </div>

        <div key={tab}>
          {tab === "general" ? (
            <Card>
              {!locked.has("categories") ? (
                <SettingRow icon={<GridIcon className="size-4.5" />} stacked title={t("default_categories")}>
                  <div className="flex flex-wrap gap-1.5">
                    {globals.categories_as_tabs.map((category) => {
                      const active = categories.includes(category);
                      return (
                        <button
                          aria-pressed={active}
                          className={`inline-flex items-center gap-1.5 rounded-full border px-3 py-1.5 text-[13px] transition-colors ${
                            active
                              ? "border-accent-strong bg-accent-soft font-medium text-accent"
                              : "border-line text-ink-2 hover:border-ink-3 hover:text-ink"
                          }`}
                          key={category}
                          onClick={() => {
                            setCategories((prev) =>
                              prev.includes(category) ? prev.filter((item) => item !== category) : [...prev, category],
                            );
                          }}
                          type="button"
                        >
                          <CategoryIcon category={category} className="size-3.5" />
                          {globals.category_labels[category] ?? category}
                        </button>
                      );
                    })}
                  </div>
                </SettingRow>
              ) : null}
              {!locked.has("language") ? (
                <SettingRow
                  description={t("what_language")}
                  icon={<LanguagesIcon className="size-4.5" />}
                  title={t("search_language")}
                >
                  <Select
                    ariaLabel={t("search_language")}
                    onChange={setLanguage}
                    options={[
                      { value: "all", label: `${t("default_language")} [all]` },
                      { value: "auto", label: t("autodetect") },
                      ...[...globals.locales]
                        .sort((a, b) => a.name.localeCompare(b.name))
                        .map((item) => ({
                          value: item.tag,
                          label: `${item.name}${item.country ? `-${item.country}` : ""} [${item.tag}] ${item.flag}`,
                        })),
                    ]}
                    value={language}
                  />
                </SettingRow>
              ) : null}
              {!locked.has("autocomplete") ? (
                <SettingRow
                  description={t("show_queries_as_you_type")}
                  icon={<SearchIcon className="size-4.5" />}
                  title={t("autocomplete")}
                >
                  <Select
                    ariaLabel={t("autocomplete")}
                    onChange={setAutocomplete}
                    options={[
                      { value: "", label: "-" },
                      ...data.autocomplete_backends.map((backend) => ({ value: backend, label: backend })),
                    ]}
                    value={autocomplete}
                  />
                </SettingRow>
              ) : null}
              {!locked.has("favicon_resolver") ? (
                <SettingRow
                  description={t("display_favicons")}
                  icon={<StarIcon className="size-4.5" />}
                  title={t("favicon_resolver")}
                >
                  <Select
                    ariaLabel={t("favicon_resolver")}
                    onChange={setFaviconResolver}
                    options={[
                      { value: "", label: "-" },
                      ...data.favicon_resolver_names.map((name) => ({ value: name, label: name })),
                    ]}
                    value={faviconResolver}
                  />
                </SettingRow>
              ) : null}
              {!locked.has("safesearch") ? (
                <SettingRow
                  description={t("filter_content")}
                  icon={<ShieldIcon className="size-4.5" />}
                  title={t("safesearch")}
                >
                  <Select
                    ariaLabel={t("safesearch")}
                    onChange={setSafesearch}
                    options={[
                      { value: "2", label: t("strict") },
                      { value: "1", label: t("moderate") },
                      { value: "0", label: t("none") },
                    ]}
                    value={safesearch}
                  />
                </SettingRow>
              ) : null}
              <SettingRow
                description={t("access_tokens")}
                icon={<KeyIcon className="size-4.5" />}
                title={t("engine_tokens")}
              >
                <input
                  autoComplete="off"
                  className="h-9 w-full rounded-xl border border-line bg-surface px-3 text-sm transition-colors hover:border-ink-3 sm:w-60"
                  onChange={(event) => {
                    setTokens(event.target.value);
                  }}
                  spellCheck={false}
                  type="text"
                  value={tokens}
                />
              </SettingRow>
              {pluginSections("general").map((plugin) => (
                <PluginRow
                  enabled={plugins[plugin.id] ?? false}
                  key={plugin.id}
                  onChange={(checked) => {
                    setPlugins((prev) => ({ ...prev, [plugin.id]: checked }));
                  }}
                  plugin={plugin}
                />
              ))}
              {!locked.has("doi_resolver") ? (
                <>
                  {pluginSections("general/doi_resolver").map((plugin) => (
                    <PluginRow
                      enabled={plugins[plugin.id] ?? false}
                      key={plugin.id}
                      onChange={(checked) => {
                        setPlugins((prev) => ({ ...prev, [plugin.id]: checked }));
                      }}
                      plugin={plugin}
                    />
                  ))}
                  <SettingRow
                    description={t("select_doi_service")}
                    icon={<BookIcon className="size-4.5" />}
                    title={t("open_access_doi_resolver")}
                  >
                    <Select
                      ariaLabel={t("open_access_doi_resolver")}
                      onChange={setDoiResolver}
                      options={Object.entries(data.doi_resolvers).map(([name]) => ({ value: name, label: name }))}
                      value={doiResolver}
                    />
                  </SettingRow>
                </>
              ) : null}
            </Card>
          ) : null}

          {tab === "ui" ? (
            <Card>
              {!locked.has("locale") ? (
                <SettingRow
                  description={t("change_layout_language")}
                  icon={<GlobeIcon className="size-4.5" />}
                  title={t("interface_language")}
                >
                  <Select
                    ariaLabel={t("interface_language")}
                    onChange={setLocale}
                    options={locales.map(([id, name]) => ({ value: id, label: name }))}
                    value={locale}
                  />
                </SettingRow>
              ) : null}
              {!locked.has("theme") ? (
                <SettingRow
                  description={t("change_layout")}
                  icon={<SparkIcon className="size-4.5" />}
                  title={t("theme")}
                >
                  <Select
                    ariaLabel={t("theme")}
                    onChange={setTheme}
                    options={data.themes.map((name) => ({
                      value: name,
                      label: name === "zjsearch" ? "ZJSearch" : name.charAt(0).toUpperCase() + name.slice(1),
                    }))}
                    value={theme}
                  />
                </SettingRow>
              ) : null}
              <SettingRow
                description={t("choose_auto")}
                icon={<MoonIcon className="size-4.5" />}
                title={t("theme_style")}
              >
                <div className="inline-flex rounded-xl border border-line bg-surface p-0.5">
                  {(
                    [
                      ["auto", cap(t("auto")), <SunIcon className="size-4" key="a" />],
                      ["light", cap(t("light")), <SunIcon className="size-4" key="l" />],
                      ["dark", cap(t("dark")), <MoonIcon className="size-4" key="d" />],
                      ["black", cap(t("black")), <MoonIcon className="size-4" key="b" />],
                    ] as const
                  ).map(([value, label, icon]) => (
                    <button
                      aria-pressed={themeStyle === value}
                      className={`inline-flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-[13px] transition-colors ${
                        themeStyle === value
                          ? "bg-accent-strong font-medium text-accent-contrast"
                          : "text-ink-2 hover:text-ink"
                      }`}
                      key={value}
                      onClick={() => {
                        setThemeStyle(value);
                        applyThemeStyle(value as ThemeStyle);
                      }}
                      title={label}
                      type="button"
                    >
                      {icon}
                      <span className="hidden md:inline">{label}</span>
                    </button>
                  ))}
                </div>
              </SettingRow>
              {!locked.has("center_alignment") ? (
                <SettingRow
                  description={t("center_alignment_desc")}
                  icon={<CenterIcon className="size-4.5" />}
                  title={t("center_alignment")}
                >
                  <Switch
                    checked={centerAlignment}
                    label={t("center_alignment")}
                    onChange={(value) => {
                      setCenterAlignment(value);
                      applyCenterAlignment(value);
                    }}
                  />
                </SettingRow>
              ) : null}
              {!locked.has("results_on_new_tab") ? (
                <SettingRow
                  description={t("open_result_new_tabs")}
                  icon={<ExternalLinkIcon className="size-4.5" />}
                  title={t("results_in_new_tabs")}
                >
                  <Switch checked={resultsOnNewTab} label={t("results_in_new_tabs")} onChange={setResultsOnNewTab} />
                </SettingRow>
              ) : null}
              {!locked.has("search_on_category_select") ? (
                <SettingRow
                  description={t("search_on_category_select_desc")}
                  icon={<CheckIcon className="size-4.5" />}
                  title={t("search_on_category_select")}
                >
                  <Switch
                    checked={searchOnCategorySelect}
                    label={t("search_on_category_select")}
                    onChange={setSearchOnCategorySelect}
                  />
                </SettingRow>
              ) : null}
              <SettingRow
                description={t("hotkeys_desc")}
                icon={<KeyboardIcon className="size-4.5" />}
                title={t("hotkeys")}
              >
                <Select
                  ariaLabel={t("hotkeys")}
                  onChange={setHotkeys}
                  options={[
                    { value: "default", label: "SearXNG" },
                    { value: "vim", label: t("hotkeys_vim") },
                  ]}
                  value={hotkeys}
                />
              </SettingRow>
              <SettingRow
                description={t("change_url_formatting")}
                icon={<LinkIcon className="size-4.5" />}
                title={t("url_formatting")}
              >
                <Select
                  ariaLabel={t("url_formatting")}
                  onChange={setUrlFormatting}
                  options={[
                    { value: "pretty", label: t("pretty") },
                    { value: "full", label: t("full") },
                    { value: "host", label: t("host") },
                  ]}
                  value={urlFormatting}
                />
              </SettingRow>
              {pluginSections("ui").map((plugin) => (
                <PluginRow
                  enabled={plugins[plugin.id] ?? false}
                  key={plugin.id}
                  onChange={(checked) => {
                    setPlugins((prev) => ({ ...prev, [plugin.id]: checked }));
                  }}
                  plugin={plugin}
                />
              ))}
            </Card>
          ) : null}

          {tab === "privacy" ? (
            <Card>
              {!locked.has("method") ? (
                <SettingRow
                  description={t("change_forms_submit")}
                  icon={<SwapIcon className="size-4.5" />}
                  title={t("http_method")}
                >
                  <div className="inline-flex rounded-xl border border-line bg-surface p-0.5">
                    {(["POST", "GET"] as const).map((value) => (
                      <button
                        aria-pressed={method === value}
                        className={`rounded-lg px-4 py-1.5 text-[13px] transition-colors ${
                          method === value
                            ? "bg-accent-strong font-medium text-accent-contrast"
                            : "text-ink-2 hover:text-ink"
                        }`}
                        key={value}
                        onClick={() => {
                          setMethod(value);
                        }}
                        type="button"
                      >
                        {value}
                      </button>
                    ))}
                  </div>
                </SettingRow>
              ) : null}
              {!locked.has("image_proxy") ? (
                <SettingRow
                  description={t("proxy_images")}
                  icon={<ImageIcon className="size-4.5" />}
                  title={t("image_proxy")}
                >
                  <Switch checked={imageProxy} label={t("image_proxy")} onChange={setImageProxy} />
                </SettingRow>
              ) : null}
              {!locked.has("query_in_title") ? (
                <SettingRow
                  description={t("query_in_title_desc")}
                  icon={<TerminalIcon className="size-4.5" />}
                  title={t("query_in_title")}
                >
                  <Switch checked={queryInTitle} label={t("query_in_title")} onChange={setQueryInTitle} />
                </SettingRow>
              ) : null}
              {pluginSections("privacy").map((plugin) => (
                <PluginRow
                  enabled={plugins[plugin.id] ?? false}
                  key={plugin.id}
                  onChange={(checked) => {
                    setPlugins((prev) => ({ ...prev, [plugin.id]: checked }));
                  }}
                  plugin={plugin}
                />
              ))}
            </Card>
          ) : null}

          {tab === "engines" ? (
            <div className="space-y-4">
              <p className="flex items-center gap-2 text-sm text-ink-2">
                <GridIcon className="size-4 text-ink-3" />
                {t("currently_used_engines")}
              </p>
              <div className="flex flex-wrap gap-1.5">
                {data.engine_tabs.map((tabInfo, index) => (
                  <button
                    className={`inline-flex items-center gap-1.5 rounded-full px-3 py-1.5 text-[13px] transition-colors ${
                      index === engineTab
                        ? "bg-accent-soft font-medium text-accent"
                        : "text-ink-2 hover:bg-surface-2 hover:text-ink"
                    }`}
                    key={tabInfo.category}
                    onClick={() => {
                      setEngineTab(index);
                    }}
                    type="button"
                  >
                    <CategoryIcon category={tabInfo.category} className="size-3.5" />
                    {tabInfo.label}
                  </button>
                ))}
              </div>
              {currentEngineTab ? (
                <EnginesTab
                  enabled={engines}
                  setAll={setAll}
                  showMetrics={showMetrics}
                  tab={currentEngineTab}
                  toggleEngine={toggleEngine}
                />
              ) : null}
            </div>
          ) : null}

          {tab === "query" ? (
            <div className="overflow-x-auto rounded-2xl border border-line bg-surface">
              <table className="w-full min-w-[560px] text-left text-xs">
                <thead className="bg-surface-2 text-ink-3">
                  <tr>
                    <th className="px-4 py-3 font-medium">{t("allow")}</th>
                    <th className="px-4 py-3 font-medium">{t("keywords")}</th>
                    <th className="px-4 py-3 font-medium">{t("name")}</th>
                    <th className="px-4 py-3 font-medium">{t("description")}</th>
                    <th className="px-4 py-3 font-medium">{t("examples")}</th>
                  </tr>
                </thead>
                <tbody>
                  <tr className="bg-surface-2/60">
                    <td className="px-4 py-2.5 font-medium text-ink-2" colSpan={5}>
                      {t("instant_answer_modules")}
                    </td>
                  </tr>
                  {data.answerers.map((answerer) => (
                    <tr className="border-t border-line" key={answerer.name}>
                      <td className="px-4 py-3 text-center text-ink-3">–</td>
                      <td className="px-4 py-3">
                        <code className="rounded bg-surface-2 px-1.5 py-0.5">{answerer.keywords.join(", ")}</code>
                      </td>
                      <td className="px-4 py-2.5 font-medium">{answerer.name}</td>
                      <td className="px-4 py-2.5 text-ink-2">{answerer.description}</td>
                      <td className="px-4 py-2.5 text-ink-2">{answerer.examples.join(", ")}</td>
                    </tr>
                  ))}
                  <tr className="bg-surface-2/60">
                    <td className="px-4 py-2.5 font-medium text-ink-2" colSpan={5}>
                      {t("plugins_list")}
                    </td>
                  </tr>
                  {data.plugins
                    .filter((plugin) => plugin.section === "query")
                    .map((plugin) => (
                      <tr className="border-t border-line" key={plugin.id}>
                        <td className="px-4 py-3">
                          <Switch
                            checked={plugins[plugin.id] ?? false}
                            label={plugin.name}
                            onChange={(value) => {
                              setPlugins((prev) => ({ ...prev, [plugin.id]: value }));
                            }}
                          />
                        </td>
                        <td className="px-4 py-3">
                          <code className="rounded bg-surface-2 px-1.5 py-0.5">{plugin.keywords.join(", ")}</code>
                        </td>
                        <td className="px-4 py-2.5 font-medium">{plugin.name}</td>
                        <td className="px-4 py-2.5 text-ink-2">{plugin.description}</td>
                        <td className="px-4 py-2.5 text-ink-2">{plugin.examples.join(", ")}</td>
                      </tr>
                    ))}
                </tbody>
              </table>
            </div>
          ) : null}

          {tab === "cookies" ? (
            <>
              <Card>
                <SettingRow
                  description={t("cookies_list_desc")}
                  icon={<CookieIcon className="size-4.5" />}
                  title={t("cookies")}
                >
                  <span className="text-xs text-ink-3">
                    {data.cookies.length > 0 ? `${data.cookies.length}` : t("no_cookies")}
                  </span>
                </SettingRow>
              </Card>
              {data.cookies.length > 0 ? (
                <div className="mt-4 overflow-hidden rounded-2xl border border-line bg-surface">
                  <table className="w-full text-left text-xs">
                    <thead className="bg-surface-2 text-ink-3">
                      <tr>
                        <th className="px-4 py-2 font-medium">{t("cookie_name")}</th>
                        <th className="px-4 py-2 font-medium">{t("value")}</th>
                      </tr>
                    </thead>
                    <tbody>
                      {data.cookies.map((cookie) => (
                        <tr className="border-t border-line" key={cookie.name}>
                          <td className="px-4 py-1.5 font-mono">{cookie.name}</td>
                          <td className="px-4 py-1.5 break-all">{cookie.value}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ) : null}

              <Card>
                <div className="px-4 py-4 sm:px-5">
                  <h4 className="flex items-center gap-2 text-sm font-semibold text-ink">
                    <LinkIcon className="size-4 text-accent" />
                    {t("search_url_of_prefs")}
                  </h4>
                  <pre
                    className="mt-2 min-w-0 overflow-x-auto rounded-xl bg-surface-2 p-2.5 font-mono text-[11px] break-all whitespace-pre-wrap text-ink-2"
                    dir="ltr"
                  >
                    {shareOrigin}/?preferences={data.preferences_url_params}&amp;q=%s
                  </pre>
                  <p className="mt-1.5 text-xs text-ink-3">{t("prefs_url_privacy_note")}</p>
                </div>
                <div className="px-4 py-4 sm:px-5">
                  <h4 className="flex items-center gap-2 text-sm font-semibold text-ink">
                    <ExternalLinkIcon className="size-4 text-accent" />
                    {t("url_to_restore")}
                  </h4>
                  <pre
                    className="mt-2 min-w-0 overflow-x-auto rounded-xl bg-surface-2 p-2.5 font-mono text-[11px] break-all whitespace-pre-wrap text-ink-2"
                    dir="ltr"
                  >
                    {shareOrigin}/preferences?preferences={data.preferences_url_params}
                  </pre>
                  <p className="mt-1.5 text-xs text-ink-3">{t("url_restore_desc")}</p>
                </div>
                <div className="px-4 py-4 sm:px-5">
                  <h4 className="flex items-center gap-2 text-sm font-semibold text-ink">
                    <KeyIcon className="size-4 text-accent" />
                    {t("copy_prefs_hash")}
                  </h4>
                  <div className="mt-2 flex items-start gap-2">
                    <pre
                      className="min-w-0 flex-1 overflow-x-auto rounded-xl bg-surface-2 p-2.5 font-mono text-[11px] break-all whitespace-pre-wrap text-ink-2"
                      dir="ltr"
                    >
                      {data.preferences_url_params}
                    </pre>
                    <button
                      className="inline-flex shrink-0 items-center gap-1.5 rounded-xl bg-surface-2 px-3 py-2 text-xs text-ink-2 transition-colors hover:text-ink"
                      onClick={() => {
                        void navigator.clipboard.writeText(data.preferences_url_params).catch(() => {});
                      }}
                      type="button"
                    >
                      {t("copy")}
                    </button>
                  </div>
                </div>
                <SettingRow icon={<RefreshIcon className="size-4.5" />} stacked title={t("insert_prefs_hash")}>
                  <input
                    className="h-9 w-full rounded-xl border border-line bg-surface px-3 text-sm transition-colors hover:border-ink-3"
                    onChange={(event) => {
                      setPastedHash(event.target.value);
                    }}
                    placeholder={t("prefs_hash")}
                    type="text"
                    value={pastedHash}
                  />
                </SettingRow>
              </Card>
            </>
          ) : null}
        </div>

        <div className="mt-8 flex flex-wrap items-center gap-3 text-sm">
          <Link
            className="inline-flex items-center gap-1.5 rounded-full border border-line bg-surface px-4 py-2 text-ink-2 transition-colors hover:border-danger hover:text-danger"
            href="/clear_cookies"
          >
            <RefreshIcon className="size-4" />
            {t("reset_defaults")}
          </Link>
        </div>
        <p className="mt-4 text-xs leading-relaxed text-ink-3">
          {t("settings_in_cookies")}
          <br />
          {t("cookies_convenience")}
        </p>
      </main>
    </Shell>
  );
}
