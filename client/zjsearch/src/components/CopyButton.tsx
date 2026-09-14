// SPDX-License-Identifier: Apache-2.0 WITH Commons-Clause-1.0

import { Check } from "lucide-react";
import type { ReactNode } from "react";
import { useCopyFeedback } from "@/lib/clipboard.ts";
import { useT } from "@/lib/i18n.ts";

const ACTION_PILL =
  "inline-flex items-center gap-1.5 rounded-full border px-3.5 py-1.5 text-xs font-medium transition-colors";
const ACTION_IDLE = "border-line text-ink-2 hover:border-accent hover:text-accent";
const ACTION_COPIED = "border-ok/40 text-ok";

/** Copy-to-clipboard button in the unified share-card language: a bordered
    pill that flips to a green check + 「已复制」 while the confirmation is
    up.  `className` is merged on top (for positioning / surface tweaks like
    the map's floating chip) — the pill shape and copied feedback always
    apply. */
export function CopyButton({
  value,
  label,
  icon,
  className,
}: {
  value: string;
  label?: string;
  icon?: ReactNode;
  className?: string;
}) {
  const t = useT();
  const { copy, isCopied } = useCopyFeedback();
  const copied = isCopied(value);
  return (
    <button
      className={`${ACTION_PILL} ${copied ? ACTION_COPIED : ACTION_IDLE} ${className ?? ""}`}
      onClick={() => {
        copy(value);
      }}
      type="button"
    >
      {copied ? <Check className="size-3 shrink-0" /> : icon}
      {copied ? t("copied") : (label ?? t("copy"))}
    </button>
  );
}

/** Click-to-copy wrapper: the content itself is the copy trigger - click
    copies `value` and a transient chip confirms.  Hover shows the copy hint
    (accent) so the affordance is discoverable without costing layout space;
    a completed copy turns the chip green. */
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
  const { copy, isCopied } = useCopyFeedback();
  const copied = isCopied(value);
  return (
    <div
      aria-label={t("copy")}
      className={`group/copy relative cursor-pointer ${className ?? ""}`}
      onClick={() => {
        copy(value);
      }}
      onKeyDown={(event) => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          copy(value);
        }
      }}
      role="button"
      tabIndex={0}
    >
      {children}
      <span
        aria-hidden="true"
        className={`absolute end-1.5 top-1.5 inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-[11px] transition-colors ${
          copied ? "bg-ok/10 text-ok opacity-100" : "bg-accent-soft text-accent opacity-0 group-hover/copy:opacity-100"
        }`}
      >
        {copied ? <Check className="size-3 shrink-0" /> : null}
        {copied ? t("copied") : t("copy")}
      </span>
    </div>
  );
}
