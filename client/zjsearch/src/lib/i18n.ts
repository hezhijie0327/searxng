// SPDX-License-Identifier: Apache-2.0 WITH Commons-Clause-1.0

/**
 * ZJSearch maintains its own UI string catalog instead of relying on the
 * upstream searx/translations catalogs: English is the source, and every
 * other locale falls back to English key by key. Strings are looked up by
 * key with useT() / t("key").
 *
 * Adding a language takes exactly two edits:
 *   1. create ./i18n/<tag>.ts exporting `Record<StringKey, string>` (a
 *      Partial is fine — missing keys render English),
 *   2. register it in CATALOGS below and teach themeLocaleTag() to resolve
 *      the browser locale tags onto it.
 *
 * The catalogs themselves live in ./i18n/; StringKey keeps call sites and
 * translations honest at compile time.
 */

import { createContext, useContext } from "react";
import { EN, type StringKey } from "@/lib/i18n/en.ts";
import { ZH_CN } from "@/lib/i18n/zh-CN.ts";

export type { StringKey };
export type Translate = (key: StringKey) => string;

/** Catalogs keyed by their locale tag; the catalog file name IS the tag. */
const CATALOGS: Record<string, Partial<Record<StringKey, string>>> = {
  en: EN,
  "zh-CN": ZH_CN,
};

type CatalogTag = keyof typeof CATALOGS;

/** Resolve any BCP-47 tag the server may send onto a catalog we ship.
    Simplified Chinese only (zh-CN, zh-Hans-CN, zh); Traditional stays
    English until someone contributes it. */
function themeLocaleTag(locale: string): CatalogTag {
  const tag = locale.toLowerCase();
  if (tag.startsWith("zh") && !tag.includes("hant")) {
    return "zh-CN";
  }
  return "en";
}

export const I18nContext = createContext<string>("en");

export function useT(): Translate {
  const locale = useContext(I18nContext);
  const catalog = CATALOGS[themeLocaleTag(locale)] ?? EN;
  return (key: StringKey) => catalog[key] ?? EN[key] ?? key;
}
