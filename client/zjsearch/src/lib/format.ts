// SPDX-License-Identifier: AGPL-3.0-or-later

/** Client side formatting helpers (dates, durations, response times). */

const DAY_MS = 86_400_000;

/** Relative date for recent timestamps, locale date otherwise. */
export function formatDate(iso: string): string {
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) {
    return iso;
  }
  const diff = Date.now() - date.getTime();
  const rtf = new Intl.RelativeTimeFormat(undefined, { numeric: "auto" });
  if (diff < DAY_MS) {
    return rtf.format(-Math.round(diff / 3_600_000), "hour");
  }
  if (diff < 30 * DAY_MS) {
    return rtf.format(-Math.round(diff / DAY_MS), "day");
  }
  if (diff < 365 * DAY_MS) {
    return rtf.format(-Math.round(diff / (30 * DAY_MS)), "month");
  }
  return date.toLocaleDateString(undefined, { year: "numeric", month: "short", day: "numeric" });
}

/** Video/audio duration: passthrough display string or seconds -> h:mm:ss. */
export function formatLength(lengthDisplay: string | undefined, lengthSeconds: number | undefined): string | null {
  if (lengthDisplay) {
    return lengthDisplay;
  }
  if (lengthSeconds === undefined || lengthSeconds <= 0) {
    return null;
  }
  const total = Math.round(lengthSeconds);
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  const seconds = total % 60;
  const mm = hours > 0 || minutes > 0 ? String(minutes).padStart(2, "0") : "0";
  const ss = String(seconds).padStart(2, "0");
  return hours > 0 ? `${hours}:${mm}:${ss}` : `${mm}:${ss}`;
}

export function formatSeconds(value: number | null | undefined): string {
  if (value === null || value === undefined) {
    return "";
  }
  return `${Math.round(value * 100) / 100}`;
}
