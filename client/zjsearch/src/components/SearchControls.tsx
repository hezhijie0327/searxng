// SPDX-License-Identifier: AGPL-3.0-or-later

import { useEffect, useState } from "react";
import { useT } from "../lib/i18n.ts";
import { useSettings } from "../lib/settings.ts";
import type { GlobalData, SearchPageData } from "../lib/types.ts";
import type { DropdownOption } from "./Dropdown.tsx";
import { Dropdown } from "./Dropdown.tsx";
import { CategoryIcon } from "./icons.tsx";

interface CategoryTabsProps {
  globals: GlobalData;
  selected: string[];
  /** multi-selection changes (toggle) must reach the parent so that
      query submits (Enter / search box) use the up-to-date selection */
  onSelectionChange?: (categories: string[]) => void;
  onSearch: (categories: string[]) => void;
  /** wrap onto multiple lines (index hero) instead of scrolling one row */
  wrap?: boolean;
}

/**
 * Category pills.  With `search_on_category_select` a plain click immediately
 * searches the clicked category, shift+click toggles multi-selection.
 * Otherwise it behaves like toggling checkboxes and the magnifier submits.
 */
export function CategoryTabs({ globals, selected, onSelectionChange, onSearch, wrap = false }: CategoryTabsProps) {
  const t = useT();
  const settings = useSettings();

  const toggle = (category: string) => {
    const next = selected.includes(category) ? selected.filter((item) => item !== category) : [...selected, category];
    onSelectionChange?.(next.length > 0 ? next : [globals.default_category]);
  };

  const onClick = (category: string, event: React.MouseEvent) => {
    if (event.shiftKey || !settings.search_on_category_select) {
      toggle(category);
      return;
    }
    onSearch([category]);
  };

  const tabs = globals.categories_as_tabs.length > 0 ? globals.categories_as_tabs : globals.categories;

  return (
    <div
      className={`flex items-center gap-0.5 py-1 ${
        wrap
          ? "flex-wrap justify-center gap-y-0.5"
          : "overflow-x-auto [scrollbar-width:none] [&::-webkit-scrollbar]:hidden"
      }`}
    >
      {tabs.map((category) => {
        const isSelected = selected.includes(category);
        return (
          <button
            aria-pressed={isSelected}
            className={`relative shrink-0 px-3.5 py-2 text-[13.5px] transition-colors ${
              isSelected ? "font-medium text-accent" : "text-ink-2 hover:text-ink"
            }`}
            key={category}
            onClick={(event) => {
              onClick(category, event);
            }}
            title={settings.search_on_category_select ? undefined : t("search")}
            type="button"
          >
            <span>{globals.category_labels[category] ?? category}</span>
            <span
              aria-hidden="true"
              className={`absolute inset-x-3 -bottom-0.5 h-0.5 rounded-full transition-opacity ${
                isSelected ? "bg-accent-strong opacity-100" : "opacity-0"
              }`}
            />
          </button>
        );
      })}
    </div>
  );
}

export interface FilterValues {
  language: string;
  time_range: string;
  safesearch: number;
  search_language?: string;
}

function SelectField({
  label,
  value,
  options,
  onChange,
}: {
  label: string;
  value: string;
  options: DropdownOption[];
  onChange: (value: string) => void;
}) {
  return <Dropdown ariaLabel={label} onChange={onChange} options={options} value={value} />;
}

export function SearchFilters({
  globals,
  values,
  onChange,
}: {
  globals: GlobalData;
  values: FilterValues;
  onChange: (next: Partial<FilterValues>) => void;
}) {
  const t = useT();
  const locales = [...globals.locales].sort((a, b) => a.name.localeCompare(b.name));

  return (
    <div className="flex flex-wrap items-center gap-1">
      <SelectField
        label={t("search_language")}
        onChange={(language) => onChange({ language })}
        options={[
          { value: "all", label: `${t("default_language")} [all]` },
          {
            value: "auto",
            label: `${t("autodetect")}${values.search_language ? ` (${values.search_language})` : ""}`,
          },
          ...locales.map((locale) => ({
            value: locale.tag,
            label: `${locale.name}${locale.country ? `-${locale.country}` : ""} [${locale.tag}] ${locale.flag}`,
          })),
        ]}
        value={values.language}
      />

      <SelectField
        label={t("time_range")}
        onChange={(time_range) => {
          onChange({ time_range });
        }}
        options={[
          { value: "", label: t("anytime") },
          { value: "day", label: t("last_day") },
          { value: "week", label: t("last_week") },
          { value: "month", label: t("last_month") },
          { value: "year", label: t("last_year") },
        ]}
        value={values.time_range}
      />

      <SelectField
        label={t("safesearch")}
        onChange={(value) => {
          onChange({ safesearch: Number(value) });
        }}
        options={[
          { value: "2", label: `${t("safesearch")}: ${t("strict")}` },
          { value: "1", label: `${t("safesearch")}: ${t("moderate")}` },
          { value: "0", label: `${t("safesearch")}: ${t("none")}` },
        ]}
        value={String(values.safesearch)}
      />
    </div>
  );
}

/** Filter values for a page that has no search payload yet (index page). */
export function defaultFilterValues(globals: GlobalData): FilterValues {
  return { language: globals.language || "all", time_range: "", safesearch: globals.safesearch };
}

export function filterValuesFromResults(data: SearchPageData): FilterValues {
  return {
    language: data.current_language || "all",
    time_range: data.time_range || "",
    safesearch: data.globals.safesearch,
    search_language: data.search_language,
  };
}
