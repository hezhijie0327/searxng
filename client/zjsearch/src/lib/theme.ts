// SPDX-License-Identifier: Apache-2.0 WITH Commons-Clause-1.0

/** Light/dark theme handling shared by the quick toggle and the preferences UI. */

export type ThemeStyle = "auto" | "light" | "dark" | "black";

export function readThemeStyle(): ThemeStyle {
  const match = document.cookie.match(/(?:^|; *)simple_style=(\w+)/);
  const value = match?.[1];
  if (value === "light" || value === "dark" || value === "black") {
    return value;
  }
  return "auto";
}

export function applyThemeStyle(style: ThemeStyle) {
  const dark =
    style === "dark" ||
    style === "black" ||
    (style === "auto" && window.matchMedia("(prefers-color-scheme: dark)").matches);
  const root = document.documentElement;
  root.classList.toggle("dark", dark);
  root.classList.toggle("black", style === "black");
}

export function applyCenterAlignment(on: boolean) {
  document.documentElement.classList.toggle("centered", on);
}

/** Live-follow system theme switches while the page is open (auto mode). */
export function watchSystemTheme() {
  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => {
    if (readThemeStyle() === "auto") {
      applyThemeStyle("auto");
    }
  });
}
