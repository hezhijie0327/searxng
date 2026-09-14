// SPDX-License-Identifier: Apache-2.0 WITH Commons-Clause-1.0

import {
  AlignCenterVertical,
  ExternalLink,
  Globe,
  Keyboard,
  Link as LinkIcon,
  Monitor,
  Moon,
  Search,
  Sun,
} from "lucide-react";
import { useT } from "@/lib/i18n.ts";
import type { ThemeStyle } from "@/lib/theme.ts";
import { applyCenterAlignment, applyThemeStyle } from "@/lib/theme.ts";
import type { PreferencesPageData } from "@/lib/types.ts";
import { Card, cap, PluginRow, Select, SettingRow, Switch } from "@/pages/preferences/parts.tsx";
import type { PreferencesForm } from "@/pages/preferences/usePreferencesForm.ts";

export function UiTab({
  data,
  form,
  locked,
}: {
  data: PreferencesPageData;
  form: PreferencesForm;
  locked: Set<string>;
}) {
  const t = useT();
  const locales = Object.entries(data.locales).sort((a, b) => a[1].localeCompare(b[1]));
  return (
    <Card>
      {!locked.has("locale") ? (
        <SettingRow
          description={t("change_layout_language")}
          icon={<Globe className="size-4.5" />}
          title={t("interface_language")}
        >
          <Select
            ariaLabel={t("interface_language")}
            onChange={form.setLocale}
            options={locales.map(([id, name]) => ({ value: id, label: name }))}
            value={form.locale}
          />
        </SettingRow>
      ) : null}
      {!locked.has("theme") ? (
        <SettingRow description={t("change_layout")} icon={<Monitor className="size-4.5" />} title={t("theme")}>
          <Select
            ariaLabel={t("theme")}
            onChange={form.setTheme}
            options={data.themes.map((name) => ({
              value: name,
              label: name === "zjsearch" ? "ZJSearch" : name.charAt(0).toUpperCase() + name.slice(1),
            }))}
            value={form.theme}
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
              aria-pressed={form.themeStyle === value}
              className={`inline-flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-[13px] transition-colors ${
                form.themeStyle === value
                  ? "bg-accent-strong font-medium text-accent-contrast"
                  : "text-ink-2 hover:text-ink"
              }`}
              key={value}
              onClick={() => {
                form.setThemeStyle(value);
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
            checked={form.centerAlignment}
            label={t("center_alignment")}
            onChange={(value) => {
              form.setCenterAlignment(value);
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
          <Switch checked={form.resultsOnNewTab} label={t("results_in_new_tabs")} onChange={form.setResultsOnNewTab} />
        </SettingRow>
      ) : null}
      {!locked.has("search_on_category_select") ? (
        <SettingRow
          description={t("search_on_category_select_desc")}
          icon={<Search className="size-4.5" />}
          title={t("search_on_category_select")}
        >
          <Switch
            checked={form.searchOnCategorySelect}
            label={t("search_on_category_select")}
            onChange={form.setSearchOnCategorySelect}
          />
        </SettingRow>
      ) : null}
      <SettingRow description={t("hotkeys_desc")} icon={<Keyboard className="size-4.5" />} title={t("hotkeys")}>
        <Select
          ariaLabel={t("hotkeys")}
          onChange={form.setHotkeys}
          options={[
            { value: "default", label: "SearXNG" },
            { value: "vim", label: t("hotkeys_vim") },
          ]}
          value={form.hotkeys}
        />
      </SettingRow>
      <SettingRow
        description={t("change_url_formatting")}
        icon={<LinkIcon className="size-4.5" />}
        title={t("url_formatting")}
      >
        <Select
          ariaLabel={t("url_formatting")}
          onChange={form.setUrlFormatting}
          options={[
            { value: "pretty", label: t("pretty") },
            { value: "full", label: t("full") },
            { value: "host", label: t("host") },
          ]}
          value={form.urlFormatting}
        />
      </SettingRow>
      {data.plugins
        .filter((plugin) => plugin.section === "ui")
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
    </Card>
  );
}
