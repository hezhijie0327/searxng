// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * Minimal i18n: the server translates the full UI string catalog into the
 * page payload (globals.strings), the client only looks keys up.
 */

import { createContext, useContext } from "react";

export type Translate = (key: string) => string;

const MISSING: Translate = (key) => key;

export const I18nContext = createContext<Record<string, string>>({});

export function useT(): Translate {
  const strings = useContext(I18nContext);
  if (!strings || Object.keys(strings).length === 0) {
    return MISSING;
  }
  return (key: string) => strings[key] ?? key;
}
