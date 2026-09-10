// SPDX-License-Identifier: AGPL-3.0-or-later

/** Keyboard shortcuts help dialog (opened with "?"). */

import { useEffect } from "react";
import { useT } from "../lib/i18n.ts";
import { CloseIcon } from "./icons.tsx";

const DEFAULT_ROWS: Array<[string, string]> = [
  ["↑ ↓", "Select previous / next result"],
  ["← →", "Previous / next page"],
  ["o", "Open selected result"],
  ["t", "Open selected result in a new tab"],
  ["y", "Copy URL of the selected result"],
  ["/", "Focus the search box"],
  ["?", "Show this help"],
];

const VIM_ROWS: Array<[string, string]> = [
  ["j k", "Select next / previous result"],
  ["n p", "Next / previous page"],
  ["o ⏎", "Open selected result"],
  ["v", "Open selected result in a new tab"],
  ["y", "Copy URL of the selected result"],
  ["/", "Focus the search box"],
  ["?", "Show this help"],
];

export function HelpModal({ layout, onClose }: { layout: "default" | "vim"; onClose: () => void }) {
  const t = useT();
  const rows = layout === "vim" ? VIM_ROWS : DEFAULT_ROWS;

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

  return (
    <div aria-modal="true" className="fixed inset-0 z-50 animate-fade-in" role="dialog">
      <button
        aria-label="Close"
        className="absolute inset-0 cursor-default bg-black/60"
        onClick={onClose}
        type="button"
      />
      <div className="pointer-events-none absolute inset-0 grid place-items-center p-4">
        <div className="pointer-events-auto w-full max-w-md rounded-2xl border border-line bg-surface p-5 shadow-pop animate-fade-up">
          <div className="mb-3 flex items-center justify-between">
            <h2 className="text-base font-semibold text-ink">
              {t("hotkeys")} · {layout === "vim" ? "Vim" : "Default"}
            </h2>
            <button
              aria-label={t("close")}
              className="grid size-8 place-items-center rounded-full text-ink-2 transition-colors hover:bg-surface-2 hover:text-ink"
              onClick={onClose}
              type="button"
            >
              <CloseIcon className="size-4" />
            </button>
          </div>
          <dl className="space-y-1.5">
            {rows.map(([keys, description]) => (
              <div className="flex items-center justify-between gap-4 text-sm" key={keys}>
                <dt>
                  <kbd className="rounded-md border border-line bg-surface-2 px-2 py-0.5 font-mono text-xs text-ink">
                    {keys}
                  </kbd>
                </dt>
                <dd className="text-right text-ink-2">{description}</dd>
              </div>
            ))}
          </dl>
        </div>
      </div>
    </div>
  );
}
