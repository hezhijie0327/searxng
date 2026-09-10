// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * Keyboard navigation for the results page, mirroring the upstream
 * default and vim layouts (client_settings.hotkeys).
 */

import { useEffect, useRef, useState } from "react";

export interface HotkeyTarget {
  /** select previous / next result (clamped) */
  move: (delta: number) => void;
  /** open the selected result (same tab or new tab) */
  open: (newTab: boolean) => void;
  /** copy the url of the selected result */
  yank: () => string | null;
  /** paginate */
  page: (delta: number) => void;
  /** focus the search input */
  focusSearch: () => void;
}

const TEXT_ENTRY = new Set(["INPUT", "TEXTAREA", "SELECT"]);

export function useHotkeys(layout: "default" | "vim", target: HotkeyTarget, onHelp: () => void) {
  const [selected, setSelected] = useState(-1);
  const listRef = useRef<HTMLElement | null>(null);

  const cards = () => (listRef.current ? Array.from(listRef.current.querySelectorAll("article.result")) : []);

  const move = (delta: number) => {
    const items = cards();
    if (items.length === 0) {
      return;
    }
    const next = Math.min(items.length - 1, Math.max(0, selected + delta));
    setSelected(next);
    items[next]?.scrollIntoView({ block: "center", behavior: "smooth" });
  };

  const ref = useRef(target);
  ref.current = target;

  // biome-ignore lint/correctness/useExhaustiveDependencies: handlers are read through ref.current
  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.ctrlKey || event.altKey || event.metaKey) {
        return;
      }
      const el = event.target as HTMLElement | null;
      const tag = el?.tagName ?? "";
      if (el?.isContentEditable || TEXT_ENTRY.has(tag)) {
        // inside text fields only Escape means "leave the field"
        if (event.key === "Escape" && tag === "INPUT") {
          (el as HTMLInputElement).blur();
        }
        return;
      }

      const t = ref.current;
      const vim = layout === "vim";
      const key = event.key;

      if (key === "?") {
        event.preventDefault();
        onHelp();
        return;
      }
      if (vim && key === "j") {
        event.preventDefault();
        move(1);
        return;
      }
      if (vim && key === "k") {
        event.preventDefault();
        move(-1);
        return;
      }
      if (!vim && (key === "ArrowDown" || key === "ArrowRight")) {
        event.preventDefault();
        key === "ArrowRight" ? t.page(1) : move(1);
        return;
      }
      if (!vim && (key === "ArrowUp" || key === "ArrowLeft")) {
        event.preventDefault();
        key === "ArrowLeft" ? t.page(-1) : move(-1);
        return;
      }
      if (key === "n") {
        event.preventDefault();
        t.page(1);
        return;
      }
      if (key === "p") {
        event.preventDefault();
        t.page(-1);
        return;
      }
      if (key === "o" || (vim && key === "Enter")) {
        event.preventDefault();
        t.open(false);
        return;
      }
      if (key === "t" || (vim && key === "v")) {
        event.preventDefault();
        t.open(true);
        return;
      }
      if (key === "y") {
        event.preventDefault();
        const url = t.yank();
        if (url) {
          void navigator.clipboard.writeText(url).catch(() => {});
        }
        return;
      }
      if (key === "/") {
        event.preventDefault();
        t.focusSearch();
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => {
      window.removeEventListener("keydown", onKeyDown);
    };
  }, [layout, onHelp]);

  const selectRef = useRef(selected);
  selectRef.current = selected;

  return { selected, setSelected, listRef };
}
