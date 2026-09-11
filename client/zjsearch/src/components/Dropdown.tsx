// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * Custom dropdown menu (no native <select>): Kagi-style panel with a check
 * mark on the active option, full keyboard navigation and outside-click
 * dismissal.
 */

import { type KeyboardEvent, type ReactNode, useEffect, useRef, useState } from "react";
import { CheckIcon } from "./icons.tsx";

export interface DropdownOption {
  value: string;
  label: string;
  /** optional icon rendered before the label in the menu */
  icon?: ReactNode;
}

export function Dropdown({
  value,
  options,
  onChange,
  ariaLabel,
  align = "start",
  variant = "bare",
  menuClassName = "",
  icon,
  iconOnly = false,
}: {
  value: string;
  options: DropdownOption[];
  onChange: (value: string) => void;
  ariaLabel?: string;
  /** which edge of the trigger the menu aligns to */
  align?: "start" | "end";
  /** bare: light text trigger (search filters); boxed: bordered control (preferences) */
  variant?: "bare" | "boxed";
  menuClassName?: string;
  /** optional icon shown before the label in the trigger */
  icon?: ReactNode;
  /** trigger renders only the icon (kebab-style menus) */
  iconOnly?: boolean;
}) {
  const [open, setOpen] = useState(false);
  const [active, setActive] = useState(-1);
  const rootRef = useRef<HTMLDivElement>(null);
  const current = options.find((option) => option.value === value);

  // close on outside clicks
  useEffect(() => {
    if (!open) {
      return;
    }
    const onPointerDown = (event: PointerEvent) => {
      if (!rootRef.current?.contains(event.target as Node)) {
        setOpen(false);
      }
    };
    window.addEventListener("pointerdown", onPointerDown);
    return () => {
      window.removeEventListener("pointerdown", onPointerDown);
    };
  }, [open]);

  const openMenu = () => {
    setActive(options.findIndex((option) => option.value === value));
    setOpen(true);
  };

  const pick = (index: number) => {
    const option = options[index];
    if (option) {
      onChange(option.value);
    }
    setOpen(false);
  };

  const onKeyDown = (event: KeyboardEvent) => {
    if (!open) {
      if (event.key === "ArrowDown" || event.key === "ArrowUp" || event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        openMenu();
      }
      return;
    }
    switch (event.key) {
      case "Escape": {
        setOpen(false);
        break;
      }
      case "ArrowDown": {
        event.preventDefault();
        setActive((prev) => (prev + 1) % options.length);
        break;
      }
      case "ArrowUp": {
        event.preventDefault();
        setActive((prev) => (prev <= 0 ? options.length - 1 : prev - 1));
        break;
      }
      case "Enter": {
        event.preventDefault();
        if (active >= 0) {
          pick(active);
        }
        break;
      }
    }
  };

  return (
    <div className="relative" ref={rootRef}>
      <button
        aria-expanded={open}
        aria-haspopup="listbox"
        aria-label={ariaLabel}
        className={
          iconOnly
            ? `flex size-8 items-center justify-center rounded-full transition-colors ${
                open ? "bg-surface-2 text-ink" : "text-ink-2 hover:bg-surface-2/70 hover:text-ink"
              }`
            : variant === "bare"
              ? `flex items-center gap-1.5 rounded-lg ps-3.5 pe-2.5 py-1.5 text-[13px] transition-colors ${
                  open ? "bg-surface-2 text-ink" : "text-ink-2 hover:bg-surface-2/70 hover:text-ink"
                }`
              : `flex h-9 w-full cursor-pointer items-center justify-between gap-2 rounded-xl border px-3 text-sm transition-colors ${
                  open ? "border-ink-3" : "border-line hover:border-ink-3"
                } bg-surface text-ink`
        }
        onClick={() => {
          if (open) {
            setOpen(false);
          } else {
            openMenu();
          }
        }}
        onKeyDown={onKeyDown}
        role="combobox"
        type="button"
      >
        {icon}
        {iconOnly ? null : <span className="truncate">{current?.label ?? value}</span>}
        {iconOnly ? null : (
          <svg
            aria-hidden="true"
            className={`size-3 shrink-0 opacity-70 transition-transform ${open ? "rotate-180" : ""}`}
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            viewBox="0 0 24 24"
          >
            <polyline points="6 9 12 15 18 9" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
        )}
      </button>

      {open ? (
        <ul
          aria-label={ariaLabel}
          className={`absolute top-full z-40 mt-1.5 max-h-80 min-w-44 overflow-auto rounded-2xl border border-line bg-surface py-1.5 shadow-pop animate-fade-in ${
            align === "end" ? "end-0" : "start-0"
          } ${menuClassName}`}
          role="listbox"
        >
          {options.map((option, index) => {
            const selected = option.value === value;
            return (
              <li aria-selected={selected} key={option.value} role="option">
                <button
                  className={`flex w-full items-center justify-between gap-4 px-4 py-2 text-left text-sm ${
                    index === active ? "bg-surface-2" : ""
                  }`}
                  onClick={() => {
                    pick(index);
                  }}
                  onMouseEnter={() => {
                    setActive(index);
                  }}
                  type="button"
                >
                  <span className="flex min-w-0 items-center gap-2">
                    {option.icon ? <span className="shrink-0 text-ink-3">{option.icon}</span> : null}
                    <span className={`truncate ${selected ? "font-medium text-ink" : "text-ink-2"}`}>
                      {option.label}
                    </span>
                  </span>
                  {selected ? <CheckIcon className="size-4 shrink-0 text-accent-strong" /> : null}
                </button>
              </li>
            );
          })}
        </ul>
      ) : null}
    </div>
  );
}
