// SPDX-License-Identifier: AGPL-3.0-or-later

/** Fixed-row horizontal strip with left/right paging arrows.  Section
    layouts use it to keep a predictable footprint: every result stays
    reachable by scrolling, nothing ever expands in place. */

import { type ReactNode, useCallback, useEffect, useRef, useState } from "react";
import { useT } from "../../lib/i18n.ts";
import { ChevronLeftIcon, ChevronRightIcon } from "../icons.tsx";

export function Strip({ children, rows = 1 }: { children: ReactNode[]; rows?: 1 | 2 }) {
  const t = useT();
  const trackRef = useRef<HTMLDivElement>(null);
  const [atStart, setAtStart] = useState(true);
  const [atEnd, setAtEnd] = useState(false);

  const updateEdges = useCallback(() => {
    const el = trackRef.current;
    if (!el) {
      return;
    }
    // RTL browsers run scrollLeft negative - the absolute value covers both
    const position = Math.abs(el.scrollLeft);
    const max = el.scrollWidth - el.clientWidth;
    setAtStart(position <= 4);
    setAtEnd(position >= max - 4);
  }, []);

  useEffect(() => {
    updateEdges();
    window.addEventListener("resize", updateEdges);
    return () => {
      window.removeEventListener("resize", updateEdges);
    };
  }, [updateEdges, children.length]);

  const page = (direction: 1 | -1) => {
    const el = trackRef.current;
    if (!el) {
      return;
    }
    const rtl = getComputedStyle(el).direction === "rtl";
    el.scrollBy({ left: (rtl ? -1 : 1) * direction * el.clientWidth, behavior: "smooth" });
  };

  return (
    <div className="relative">
      <div
        className={`grid auto-cols-[calc(50%-8px)] grid-flow-col gap-4 overflow-x-auto pb-1 sm:auto-cols-[calc(33.333%-10.67px)] lg:auto-cols-[calc(25%-12px)] [scrollbar-width:none] [&::-webkit-scrollbar]:hidden ${
          rows === 2 ? "grid-rows-2" : "grid-rows-1"
        }`}
        onScroll={updateEdges}
        ref={trackRef}
      >
        {children}
      </div>
      {!atStart ? (
        <button
          aria-label={t("previous_page")}
          className="absolute -start-4 top-1/2 z-10 grid size-9 -translate-y-1/2 place-items-center rounded-full border border-line bg-surface text-ink-2 shadow-pop transition-colors hover:text-accent"
          onClick={() => {
            page(-1);
          }}
          title={t("previous_page")}
          type="button"
        >
          <ChevronLeftIcon className="size-4" />
        </button>
      ) : null}
      {!atEnd ? (
        <button
          aria-label={t("next_page")}
          className="absolute -end-4 top-1/2 z-10 grid size-9 -translate-y-1/2 place-items-center rounded-full border border-line bg-surface text-ink-2 shadow-pop transition-colors hover:text-accent"
          onClick={() => {
            page(1);
          }}
          title={t("next_page")}
          type="button"
        >
          <ChevronRightIcon className="size-4" />
        </button>
      ) : null}
    </div>
  );
}
