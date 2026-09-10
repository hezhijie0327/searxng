// SPDX-License-Identifier: AGPL-3.0-or-later

/** Light/dark theme handling shared by the quick toggle and the preferences UI. */

export type ThemeStyle = "auto" | "light" | "dark";

export function readThemeStyle(): ThemeStyle {
  const match = document.cookie.match(/(?:^|; *)simple_style=(\w+)/);
  const value = match?.[1];
  if (value === "light" || value === "dark") {
    return value;
  }
  return "auto";
}

export function applyThemeStyle(style: ThemeStyle) {
  const dark = style === "dark" || (style === "auto" && window.matchMedia("(prefers-color-scheme: dark)").matches);
  document.documentElement.classList.toggle("dark", dark);
}

export function writeThemeStyle(style: ThemeStyle) {
  // "auto" means: follow the system setting, i.e. no cookie (the server shell
  // and readThemeStyle treat a missing cookie as auto).
  if (style === "auto") {
    document.cookie = "simple_style=; path=/; max-age=0; samesite=lax";
  } else {
    document.cookie = `simple_style=${style}; path=/; max-age=157680000; samesite=lax`;
  }
}
