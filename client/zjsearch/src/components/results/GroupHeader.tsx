// SPDX-License-Identifier: AGPL-3.0-or-later

import type { PointerEvent as ReactPointerEvent } from "react";
import { useT } from "../../lib/i18n.ts";
import { CategoryIcon, ChevronDownIcon, GripVerticalIcon } from "../icons.tsx";

/** Collapsible block header: category icon + translated label + result
    count; the whole header toggles the block. */
interface GripHandlers {
  onPointerDown: (event: ReactPointerEvent<HTMLSpanElement>) => void;
  onPointerUp: (event: ReactPointerEvent<HTMLSpanElement>) => void;
  onKeyDown: (event: React.KeyboardEvent<HTMLSpanElement>) => void;
}

export function GroupHeader({
  category,
  label,
  count,
  collapsed,
  onToggle,
  grip,
}: {
  category: string;
  label: string;
  count: number;
  collapsed: boolean;
  onToggle: () => void;
  grip?: GripHandlers;
}) {
  const t = useT();
  return (
    <h2 className="group flex items-center gap-1 pb-1 pt-2">
      {grip ? (
        <span
          aria-label={t("drag_reorder")}
          className="-ms-1 cursor-grab touch-none rounded p-1 text-ink-3 transition-colors hover:bg-surface-2 hover:text-ink active:cursor-grabbing"
          onKeyDown={grip.onKeyDown}
          onPointerDown={grip.onPointerDown}
          onPointerUp={grip.onPointerUp}
          role="button"
          tabIndex={0}
          title={t("drag_reorder")}
        >
          <GripVerticalIcon className="size-4" />
        </span>
      ) : null}
      <button
        aria-expanded={!collapsed}
        className="flex min-w-0 flex-1 items-center gap-1.5 text-left text-sm font-semibold text-ink"
        onClick={onToggle}
        type="button"
      >
        <CategoryIcon category={category} className="size-4 shrink-0 text-accent" />
        {label}
        <span className="font-normal text-ink-3">{count}</span>
        <ChevronDownIcon
          className={`size-4 shrink-0 text-ink-3 transition-transform ${collapsed ? "-rotate-90" : ""}`}
        />
      </button>
    </h2>
  );
}
