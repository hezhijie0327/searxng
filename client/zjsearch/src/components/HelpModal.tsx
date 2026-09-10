// SPDX-License-Identifier: AGPL-3.0-or-later

/** Keyboard shortcuts help dialog (opened with "?"). */

import { useEffect } from "react";
import { useT } from "../lib/i18n.ts";
import { CloseIcon } from "./icons.tsx";

interface HelpColumn {
  title: string;
  rows: Array<[string, string]>;
}

const SHORTCUT_ROWS: Array<[string, string]> = [
  ["? / ?", "Show / hide this help"],
  ["j ↓ · k ↑", "Focus next / previous result"],
  ["← →", "Previous / next page"],
  ["n p", "Next / previous page"],
  ["o ⏎", "Open focused result"],
  ["t v", "Open in a new tab"],
  ["y", "Copy URL of the focused result"],
  ["i", "Focus the search box"],
  ["Esc", "Close panels, blur the search box"],
];

const OPERATOR_ROWS: Array<[string, string]> = [
  ["filetype:", "Limit results to a file extension"],
  ["site:", "Limit results to a specific site"],
  ["inurl:", "Word or phrase in the URL"],
  ["intitle:", "Word or phrase in the title"],
  ['"words"', "Exact phrase"],
  ["AND", "Both terms: cats AND dogs"],
  ["OR", "Either term: cats OR dogs"],
  ["+ -", "Force include / exclude a term"],
  ["*", "Wildcard within a phrase"],
];

const BANG_ROWS: Array<[string, string]> = [
  ["!bang", "All DuckDuckGo bangs work here"],
  ["!images !i", "Image search bang"],
  ["!videos !v", "Video search bang"],
  ["!news !n", "News search bang"],
  ["!maps !m", "Map search bang"],
  ["keyword", "Open the first result"],
  ["ip", "Show your IP address"],
  ["hash md5 …", "Hash a string (enable the plugin)"],
  ["random", "Random values generator"],
];

const COLUMNS: HelpColumn[] = [
  { title: "Keyboard shortcuts", rows: SHORTCUT_ROWS },
  { title: "Search operators", rows: OPERATOR_ROWS },
  { title: "Bangs & widgets", rows: BANG_ROWS },
];

export function HelpModal({ layout, onClose }: { layout: "default" | "vim"; onClose: () => void }) {
  const t = useT();

  useEffect(() => {
    const onKeyDown = (event: globalThis.KeyboardEvent) => {
      if (event.key === "Escape") {
        onClose();
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => {
      window.removeEventListener("keydown", onKeyDown);
    };
  }, [onClose]);

  // the vim column labels differ slightly from the default ones
  const columns = COLUMNS.map((column) => {
    if (column.title !== "Keyboard shortcuts") {
      return column;
    }
    const rows: Array<[string, string]> =
      layout === "vim"
        ? [
            ["?", "Show / hide this help"],
            ["j / ↓", "Focus next result"],
            ["k / ↑", "Focus previous result"],
            ["n", "Next page"],
            ["p", "Previous page"],
            ["o ⏎", "Open focused result"],
            ["v", "Open in a new tab"],
            ["y", "Copy URL of the focused result"],
            ["i", "Focus the search box"],
          ]
        : column.rows;
    return { title: column.title, rows };
  });

  return (
    <div aria-modal="true" className="fixed inset-0 z-50 animate-fade-in" role="dialog">
      <button
        aria-label="Close"
        className="absolute inset-0 cursor-default bg-black/60"
        onClick={onClose}
        type="button"
      />
      <div className="pointer-events-none absolute inset-0 grid place-items-center p-4">
        <div className="pointer-events-auto max-h-[86dvh] w-full max-w-5xl overflow-auto rounded-2xl border border-line bg-surface p-6 shadow-pop animate-fade-up">
          <div className="mb-4 flex items-center justify-between">
            <h2 className="text-lg font-semibold text-ink">{t("hotkeys")}</h2>
            <button
              aria-label={t("close")}
              className="grid size-8 place-items-center rounded-full text-ink-2 transition-colors hover:bg-surface-2 hover:text-ink"
              onClick={onClose}
              type="button"
            >
              <CloseIcon className="size-4" />
            </button>
          </div>
          <div className="grid gap-8 md:grid-cols-3">
            {columns.map((column) => (
              <section key={column.title}>
                <h3 className="mb-2 border-b border-line pb-2 text-base font-semibold text-ink">{column.title}</h3>
                <dl className="space-y-2">
                  {column.rows.map(([keys, description]) => (
                    <div className="grid grid-cols-[7.5rem_1fr] items-baseline gap-3" key={keys}>
                      <dt className="text-right font-mono text-xs font-semibold text-accent">{keys}</dt>
                      <dd className="text-[13px] leading-relaxed text-ink-2">{description}</dd>
                    </div>
                  ))}
                </dl>
              </section>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
