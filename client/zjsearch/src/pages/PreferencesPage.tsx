// SPDX-License-Identifier: Apache-2.0 WITH Commons-Clause-1.0

import {
  AlertTriangle,
  AlignCenterVertical,
  ArrowLeftRight,
  Book,
  Check,
  Cookie,
  ExternalLink,
  Globe,
  Image,
  Key,
  Keyboard,
  Languages,
  LayoutGrid,
  Link as LinkIcon,
  Monitor,
  Moon,
  RefreshCw,
  Search,
  Shield,
  SlidersHorizontal,
  Sun,
  Tag,
  Terminal,
  X,
} from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import { Link, Shell } from "../components/Shell.tsx";
import { useT } from "../lib/i18n.ts";
import type { ThemeStyle } from "../lib/theme.ts";
import { applyCenterAlignment, applyThemeStyle } from "../lib/theme.ts";
import type { PreferencesPageData } from "../lib/types.ts";
import { EnginesTab } from "./preferences/EnginesTab.tsx";
import { Card, CategoryTab, cap, PluginRow, Select, SettingRow, Switch } from "./preferences/parts.tsx";

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
    { id: "general", label: t("general"), icon: <SlidersHorizontal className="size-3.5" /> },
    { id: "ui", label: t("user_interface"), icon: <Sun className="size-3.5" /> },
    { id: "privacy", label: t("privacy"), icon: <Shield className="size-3.5" /> },
    { id: "engines", label: t("engines"), icon: <LayoutGrid className="size-3.5" /> },
    { id: "query", label: t("special_queries"), icon: <Terminal className="size-3.5" /> },
    { id: "cookies", label: t("cookies"), icon: <Cookie className="size-3.5" /> },
  ] as const;

  return (
    <Shell embedded={embedded} globals={globals}>
      <main className="mx-auto w-full max-w-4xl flex-1 px-4 pb-20 sm:px-6">
        {!embedded ? (
          <div className="flex items-center justify-between py-6">
            <h1 className="text-2xl font-semibold tracking-tight text-ink">{t("preferences")}</h1>
            <div className="flex items-center gap-3">
              {savedAt > 0 ? (
                <span className="inline-flex items-center gap-1 text-xs text-ok animate-fade-in">
                  <Check className="size-3.5" />
                  {t("saved")}
                </span>
              ) : null}
            </div>
          </div>
        ) : savedAt > 0 ? (
          <div className="flex justify-end py-3">
            <span className="inline-flex items-center gap-1 text-xs text-ok animate-fade-in">
              <Check className="size-3.5" />
              {t("saved")}
            </span>
          </div>
        ) : null}

        {isPreview ? (
          <div className="mb-4 flex items-start gap-3 rounded-2xl border border-warning/40 bg-warning/10 p-4 text-sm text-ink-2 animate-fade-up">
            <AlertTriangle className="mt-0.5 size-4 shrink-0 text-warning" />
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
              className={`flex flex-1 items-center justify-center gap-2 whitespace-nowrap rounded-xl px-4 py-2 text-[13px] transition-colors ${
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
                <SettingRow icon={<LayoutGrid className="size-4.5" />} stacked title={t("default_categories")}>
                  <div className="flex flex-wrap items-center gap-x-1 gap-y-0.5">
                    {globals.categories_as_tabs.map((category) => (
                      <CategoryTab
                        active={categories.includes(category)}
                        category={category}
                        key={category}
                        label={globals.category_labels[category] ?? category}
                        onClick={() => {
                          setCategories((prev) =>
                            prev.includes(category) ? prev.filter((item) => item !== category) : [...prev, category],
                          );
                        }}
                      />
                    ))}
                  </div>
                </SettingRow>
              ) : null}
              {!locked.has("language") ? (
                <SettingRow
                  description={t("what_language")}
                  icon={<Languages className="size-4.5" />}
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
                  icon={<Search className="size-4.5" />}
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
                  icon={<Globe className="size-4.5" />}
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
                  icon={<Shield className="size-4.5" />}
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
                icon={<Key className="size-4.5" />}
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
                    icon={<Book className="size-4.5" />}
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
                  icon={<Globe className="size-4.5" />}
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
                <SettingRow description={t("change_layout")} icon={<Monitor className="size-4.5" />} title={t("theme")}>
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
              <SettingRow description={t("choose_auto")} icon={<Sun className="size-4.5" />} title={t("theme_style")}>
                <div className="inline-flex rounded-xl border border-line bg-surface p-0.5">
                  {(
                    [
                      ["auto", cap(t("auto")), <Sun className="size-4" key="a" />],
                      ["light", cap(t("light")), <Sun className="size-4" key="l" />],
                      ["dark", cap(t("dark")), <Moon className="size-4" key="d" />],
                      ["black", cap(t("black")), <Moon className="size-4" key="b" />],
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
                  icon={<AlignCenterVertical className="size-4.5" />}
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
                  icon={<ExternalLink className="size-4.5" />}
                  title={t("results_in_new_tabs")}
                >
                  <Switch checked={resultsOnNewTab} label={t("results_in_new_tabs")} onChange={setResultsOnNewTab} />
                </SettingRow>
              ) : null}
              {!locked.has("search_on_category_select") ? (
                <SettingRow
                  description={t("search_on_category_select_desc")}
                  icon={<Search className="size-4.5" />}
                  title={t("search_on_category_select")}
                >
                  <Switch
                    checked={searchOnCategorySelect}
                    label={t("search_on_category_select")}
                    onChange={setSearchOnCategorySelect}
                  />
                </SettingRow>
              ) : null}
              <SettingRow description={t("hotkeys_desc")} icon={<Keyboard className="size-4.5" />} title={t("hotkeys")}>
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
                  icon={<ArrowLeftRight className="size-4.5" />}
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
                  icon={<Image className="size-4.5" />}
                  title={t("image_proxy")}
                >
                  <Switch checked={imageProxy} label={t("image_proxy")} onChange={setImageProxy} />
                </SettingRow>
              ) : null}
              {!locked.has("query_in_title") ? (
                <SettingRow
                  description={t("query_in_title_desc")}
                  icon={<Tag className="size-4.5" />}
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
              <div className="flex flex-wrap items-center justify-between gap-2">
                <p className="flex items-center gap-2 text-sm text-ink-2">
                  <LayoutGrid className="size-4 text-ink-3" />
                  {t("currently_used_engines")}
                </p>
                {currentEngineTab ? (
                  <div className="flex items-center gap-1.5">
                    <button
                      className="inline-flex items-center gap-1.5 rounded-full border border-line px-3 py-1.5 text-[13px] text-ink-2 transition-colors hover:border-ok hover:text-ok"
                      onClick={() => {
                        setAll(
                          currentEngineTab.groups.flatMap((group) =>
                            group.engines.map((engine) => `${engine.name}__${currentEngineTab.category}`),
                          ),
                          true,
                        );
                      }}
                      type="button"
                    >
                      <Check className="size-3.5" />
                      {t("enable_all")}
                    </button>
                    <button
                      className="inline-flex items-center gap-1.5 rounded-full border border-line px-3 py-1.5 text-[13px] text-ink-2 transition-colors hover:border-danger hover:text-danger"
                      onClick={() => {
                        setAll(
                          currentEngineTab.groups.flatMap((group) =>
                            group.engines.map((engine) => `${engine.name}__${currentEngineTab.category}`),
                          ),
                          false,
                        );
                      }}
                      type="button"
                    >
                      <X className="size-3.5" />
                      {t("disable_all")}
                    </button>
                  </div>
                ) : null}
              </div>
              <div className="flex flex-wrap items-center gap-x-1 gap-y-0.5">
                {data.engine_tabs.map((tabInfo, index) => (
                  <CategoryTab
                    active={index === engineTab}
                    category={tabInfo.category}
                    key={tabInfo.category}
                    label={tabInfo.label}
                    onClick={() => {
                      setEngineTab(index);
                    }}
                  />
                ))}
              </div>
              {currentEngineTab ? (
                <EnginesTab
                  enabled={engines}
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
            <div className="space-y-4">
              <Card>
                <SettingRow
                  description={t("cookies_list_desc")}
                  icon={<Cookie className="size-4.5" />}
                  title={t("cookies")}
                >
                  <span className="text-xs text-ink-3">
                    {data.cookies.length > 0 ? `${data.cookies.length}` : t("no_cookies")}
                  </span>
                </SettingRow>
              </Card>
              {data.cookies.length > 0 ? (
                <div className="overflow-hidden rounded-2xl border border-line bg-surface">
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
                          <td
                            className="px-4 py-1.5 break-all"
                            title={cookie.value.length > 64 ? cookie.value : undefined}
                          >
                            {cookie.value.length > 64 ? `${cookie.value.slice(0, 64)}…` : cookie.value}
                          </td>
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
                    className="mt-2 min-w-0 overflow-x-auto rounded-xl bg-surface-2 p-2.5 font-mono text-xs break-all whitespace-pre-wrap text-ink-2"
                    dir="ltr"
                  >
                    {shareOrigin}/?preferences={data.preferences_url_params}&amp;q=%s
                  </pre>
                  <p className="mt-1.5 text-xs text-ink-3">{t("prefs_url_privacy_note")}</p>
                </div>
                <div className="px-4 py-4 sm:px-5">
                  <h4 className="flex items-center gap-2 text-sm font-semibold text-ink">
                    <ExternalLink className="size-4 text-accent" />
                    {t("url_to_restore")}
                  </h4>
                  <pre
                    className="mt-2 min-w-0 overflow-x-auto rounded-xl bg-surface-2 p-2.5 font-mono text-xs break-all whitespace-pre-wrap text-ink-2"
                    dir="ltr"
                  >
                    {shareOrigin}/preferences?preferences={data.preferences_url_params}
                  </pre>
                  <p className="mt-1.5 text-xs text-ink-3">{t("url_restore_desc")}</p>
                </div>
                <div className="px-4 py-4 sm:px-5">
                  <h4 className="flex items-center gap-2 text-sm font-semibold text-ink">
                    <Key className="size-4 text-accent" />
                    {t("copy_prefs_hash")}
                  </h4>
                  <div className="mt-2 flex items-start gap-2">
                    <pre
                      className="min-w-0 flex-1 overflow-x-auto rounded-xl bg-surface-2 p-2.5 font-mono text-xs break-all whitespace-pre-wrap text-ink-2"
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
                <SettingRow icon={<RefreshCw className="size-4.5" />} stacked title={t("insert_prefs_hash")}>
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
              <Card>
                <div className="flex flex-wrap items-center justify-between gap-3 px-4 py-4 sm:px-5">
                  <p className="min-w-0 flex-1 text-xs leading-relaxed text-ink-3">
                    {t("settings_in_cookies")}
                    <br />
                    {t("cookies_convenience")}
                  </p>
                  <Link
                    className="inline-flex shrink-0 items-center gap-1.5 rounded-full border border-line bg-surface px-4 py-2 text-[13px] text-ink-2 transition-colors hover:border-danger hover:text-danger"
                    href="/clear_cookies"
                  >
                    <RefreshCw className="size-4" />
                    {t("reset_defaults")}
                  </Link>
                </div>
              </Card>
            </div>
          ) : null}
        </div>
        <div className="mt-6 space-y-1 text-center text-xs text-ink-3">
          <p className="leading-5">
            {t("powered_by")}{" "}
            <a
              className="transition-colors hover:text-accent hover:underline"
              href={globals.git_url}
              rel="noreferrer"
              target="_blank"
            >
              SearXNG
            </a>
            {globals.version ? <span className="ms-1 opacity-70">v{globals.version}</span> : null}
          </p>
          <p className="leading-5">
            {t("license")}:{" "}
            <a
              className="transition-colors hover:text-accent hover:underline"
              href="/static/themes/zjsearch/LICENSE.txt"
              rel="noreferrer"
              target="_blank"
            >
              Apache-2.0 with Commons Clause v1.0
            </a>
          </p>
        </div>
      </main>
    </Shell>
  );
}
