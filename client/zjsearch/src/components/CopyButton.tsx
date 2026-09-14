// SPDX-License-Identifier: Apache-2.0 WITH Commons-Clause-1.0

import type { ReactNode } from "react";
import { useState } from "react";
import { useT } from "@/lib/i18n.ts";

/** Copy-to-clipboard button with a transient "copied" confirmation.
    Pass `className` to restyle (defaults to the muted surface chip). */
export function CopyButton({
  value,
  label,
  icon,
  className = "shrink-0 rounded-lg bg-surface-2 px-2.5 py-1.5 text-xs text-ink-2 transition-colors hover:text-ink",
}: {
  value: string;
  label?: string;
  icon?: ReactNode;
  className?: string;
}) {
  const t = useT();
  const [copied, setCopied] = useState(false);
  return (
    <button
      className={className}
      onClick={() => {
        void navigator.clipboard
          .writeText(value)
          .then(() => {
            setCopied(true);
            window.setTimeout(() => {
              setCopied(false);
            }, 1500);
          })
          .catch(() => {
            /* clipboard unavailable */
          });
      }}
      type="button"
    >
      {copied ? (
        t("copied")
      ) : (
        <>
          {icon}
          {label ?? t("copy")}
        </>
      )}
    </button>
  );
}

/** Click-to-copy wrapper: the content itself is the copy trigger - click
    copies `value` and a transient chip confirms.  Hover shows the copy hint
    so the affordance is discoverable without costing layout space. */
export function ClickToCopy({
  value,
  className = "",
  children,
}: {
  value: string;
  className?: string;
  children: ReactNode;
}) {
  const t = useT();
  const [copied, setCopied] = useState(false);
  const copy = () => {
    void navigator.clipboard
      .writeText(value)
      .then(() => {
        setCopied(true);
        window.setTimeout(() => setCopied(false), 1500);
      })
      .catch(() => {
        /* clipboard unavailable */
      });
  };
  return (
    <div
      aria-label={t("copy")}
      className={`group/copy relative cursor-pointer ${className ?? ""}`}
      onClick={() => {
        copy();
      }}
      onKeyDown={(event) => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          copy();
        }
      }}
      role="button"
      tabIndex={0}
    >
      {children}
      <span
        aria-hidden="true"
        className={`absolute end-1.5 top-1.5 rounded-full bg-accent-soft px-2 py-0.5 text-[11px] text-accent transition-opacity ${
          copied ? "opacity-100" : "opacity-0 group-hover/copy:opacity-100"
        }`}
      >
        {copied ? t("copied") : t("copy")}
      </span>
    </div>
  );
}
