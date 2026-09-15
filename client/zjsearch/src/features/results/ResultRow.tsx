// SPDX-License-Identifier: Apache-2.0 WITH Commons-Clause-1.0

import type { ReactNode } from "react";

/** First N rows play the entrance animation, with a per-row stagger. */
const FADE_UP_SLOTS = 12;
const MAX_STAGGER_MS = 300;

/** Wrapper for one card in a list layout: page-global hotkey index,
    hotkey-selection ring and the entrance stagger — the scaffold shared by
    every list view (grids mark their own cells instead). */
export function ResultRow({ index, selected, children }: { index: number; selected: boolean; children: ReactNode }) {
  const animate = index < FADE_UP_SLOTS;
  return (
    <div
      className={`break-inside-avoid rounded-2xl ${selected ? "bg-surface ring-1 ring-accent-strong" : ""} ${
        animate ? "animate-fade-up" : ""
      }`}
      data-hotkey-index={index}
      style={animate ? { animationDelay: `${Math.min(index * 30, MAX_STAGGER_MS)}ms` } : undefined}
    >
      {children}
    </div>
  );
}
