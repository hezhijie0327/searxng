// SPDX-License-Identifier: Apache-2.0 WITH Commons-Clause-1.0

import { Book, Globe, Key, Languages, LayoutGrid, Search, Shield } from "lucide-react";
import { categoryLabel } from "@/lib/categories.ts";
import { useT } from "@/lib/i18n.ts";
import type { PreferencesPageData } from "@/lib/types.ts";
import { Card, CategoryTab, PluginRow, Select, SettingRow } from "@/pages/preferences/parts.tsx";
import type { PreferencesForm } from "@/pages/preferences/usePreferencesForm.ts";

export function GeneralTab({
  data,
  form,
  locked,
}: {
  data: PreferencesPageData;
  form: PreferencesForm;
  locked: Set<string>;
}) {
  const t = useT();
  const globals = data.globals;
  return (
    <Card>
      {!locked.has("categories") ? (
        <SettingRow icon={<LayoutGrid className="size-4.5" />} stacked title={t("default_categories")}>
          <div className="flex flex-wrap items-center gap-x-1 gap-y-0.5">
            {/* globals.categories = tabs filtered to categories with enabled
                engines — mirrors upstream simple preferences behaviour */}
            {globals.categories.map((category) => (
              <CategoryTab
                active={form.categories.includes(category)}
                category={category}
                key={category}
                label={categoryLabel(category, t)}
                onClick={() => {
                  form.setCategories((prev) =>
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
            onChange={form.setLanguage}
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
            value={form.language}
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
            onChange={form.setAutocomplete}
            options={[
              { value: "", label: "-" },
              ...data.autocomplete_backends.map((backend) => ({ value: backend, label: backend })),
            ]}
            value={form.autocomplete}
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
            onChange={form.setFaviconResolver}
            options={[
              { value: "", label: "-" },
              ...data.favicon_resolver_names.map((name) => ({ value: name, label: name })),
            ]}
            value={form.faviconResolver}
          />
        </SettingRow>
      ) : null}
      {!locked.has("safesearch") ? (
        <SettingRow description={t("filter_content")} icon={<Shield className="size-4.5" />} title={t("safesearch")}>
          <Select
            ariaLabel={t("safesearch")}
            onChange={form.setSafesearch}
            options={[
              { value: "2", label: t("strict") },
              { value: "1", label: t("moderate") },
              { value: "0", label: t("none") },
            ]}
            value={form.safesearch}
          />
        </SettingRow>
      ) : null}
      <SettingRow description={t("access_tokens")} icon={<Key className="size-4.5" />} title={t("engine_tokens")}>
        <input
          aria-label={t("engine_tokens")}
          autoComplete="off"
          className="h-9 w-full rounded-xl border border-line bg-surface px-3 text-sm transition-colors hover:border-ink-3 sm:w-60"
          onChange={(event) => {
            form.setTokens(event.target.value);
          }}
          spellCheck={false}
          type="text"
          value={form.tokens}
        />
      </SettingRow>
      {data.plugins
        .filter((plugin) => plugin.section === "general")
        .map((plugin) => (
          <PluginRow
            enabled={form.plugins[plugin.id] ?? false}
            key={plugin.id}
            onChange={(checked) => {
              form.setPluginEnabled(plugin.id, checked);
            }}
            plugin={plugin}
          />
        ))}
      {!locked.has("doi_resolver") ? (
        <>
          {data.plugins
            .filter((plugin) => plugin.section === "general/doi_resolver")
            .map((plugin) => (
              <PluginRow
                enabled={form.plugins[plugin.id] ?? false}
                key={plugin.id}
                onChange={(checked) => {
                  form.setPluginEnabled(plugin.id, checked);
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
              onChange={form.setDoiResolver}
              options={Object.entries(data.doi_resolvers).map(([name]) => ({ value: name, label: name }))}
              value={form.doiResolver}
            />
          </SettingRow>
        </>
      ) : null}
    </Card>
  );
}
