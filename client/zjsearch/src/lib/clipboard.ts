// SPDX-License-Identifier: Apache-2.0 WITH Commons-Clause-1.0

import { useCallback, useEffect, useRef, useState } from "react";

/** Shared click-to-copy feedback: writes `value` to the clipboard and holds
    a transient "copied" confirmation (1.5 s, matching the share card) that
    callers render through `isCopied(value)`. */
export function useCopyFeedback(timeoutMs = 1500) {
  const [copiedValue, setCopiedValue] = useState<string | null>(null);
  const timer = useRef<number | null>(null);
  useEffect(
    () => () => {
      if (timer.current !== null) {
        window.clearTimeout(timer.current);
      }
    },
    [],
  );
  const copy = useCallback(
    (value: string) => {
      void navigator.clipboard
        .writeText(value)
        .then(() => {
          setCopiedValue(value);
          if (timer.current !== null) {
            window.clearTimeout(timer.current);
          }
          timer.current = window.setTimeout(() => setCopiedValue(null), timeoutMs);
        })
        .catch(() => {
          /* clipboard unavailable */
        });
    },
    [timeoutMs],
  );
  const isCopied = useCallback((value: string) => copiedValue === value, [copiedValue]);
  return { copy, isCopied };
}
