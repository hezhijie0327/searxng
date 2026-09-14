// SPDX-License-Identifier: Apache-2.0 WITH Commons-Clause-1.0

/** Raw clipboard write shared by every copy path (buttons, wrappers,
    hotkeys).  Copy confirmation is rendered by `flashToast` in
    `lib/toast.ts` — one green 「已复制」 pill app-wide. */
export function writeClipboard(value: string): Promise<boolean> {
  return navigator.clipboard
    .writeText(value)
    .then(() => true)
    .catch(() => false);
}
