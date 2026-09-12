// SPDX-License-Identifier: AGPL-3.0-or-later

import { useState } from "react";
import { useT } from "../lib/i18n.ts";

/** Copy-to-clipboard button with a transient "copied" confirmation.
    Pass `className` to restyle (defaults to the muted surface chip). */
export function CopyButton({
  value,
  label,
  className = "shrink-0 rounded-lg bg-surface-2 px-2.5 py-1.5 text-xs text-ink-2 transition-colors hover:text-ink",
}: {
  value: string;
  label?: string;
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
      {copied ? t("copied") : (label ?? t("copy"))}
    </button>
  );
}
