// SPDX-License-Identifier: Apache-2.0 WITH Commons-Clause-1.0

import { ClickToCopy } from "@/components/CopyButton.tsx";
import { newTabLinkProps } from "@/lib/link.ts";
import { useSettings } from "@/lib/settings.ts";
import type { AnswerData } from "@/lib/types.ts";

/** Special-query answers (random, statistics, hash, self-info, time zone)
    carry a structured *data* payload emitted by their plugins; the layouts
    below render strictly from those fields — no parsing of the localized
    *answer* text — and unknown payloads fall back to plain text. */
export function LegacyAnswer({ answer }: { answer: Extract<AnswerData, { template: "answer/legacy.html" }> }) {
  const settings = useSettings();
  const text = answer.answer;
  const data = answer.data;
  let hostname = "";
  if (answer.url) {
    try {
      hostname = new URL(answer.url).hostname;
    } catch {
      hostname = answer.url;
    }
  }
  if (data?.kind === "hash") {
    return (
      <div>
        <div className="flex items-center justify-between gap-3">
          <span className="rounded-full bg-surface-2 px-2 py-0.5 font-mono text-xs text-ink-2" dir="ltr">
            {data.algo}
          </span>
        </div>
        <ClickToCopy className="mt-2" value={data.digest}>
          <p className="break-all font-mono text-sm text-ink-2" dir="ltr">
            {data.digest}
          </p>
        </ClickToCopy>
      </div>
    );
  }
  if (data?.kind === "stats") {
    return (
      <div>
        <p className="truncate text-xs text-ink-3" dir="ltr">
          <span className="font-mono font-medium text-accent">{data.func}</span>({data.args})
        </p>
        <div className="mt-1 flex items-center justify-between gap-3">
          <ClickToCopy className="mt-1" value={data.result}>
            <p className="text-2xl font-semibold text-ink" dir="ltr">
              {data.result}
            </p>
          </ClickToCopy>
        </div>
      </div>
    );
  }
  if (data?.kind === "time") {
    return (
      <div className="flex items-center justify-between gap-3">
        <div className="min-w-0">
          {data.zone ? (
            <p className="truncate font-mono text-xs text-ink-3" dir="ltr">
              {data.zone}
            </p>
          ) : null}
          <p className="mt-1 text-xl font-medium text-ink" dir="auto">
            {data.time}
          </p>
        </div>
        {data.abbr ? (
          <span className="shrink-0 rounded-full bg-surface-2 px-2 py-0.5 font-mono text-xs text-ink-2">
            {data.abbr}
          </span>
        ) : null}
      </div>
    );
  }
  if (data?.kind === "self") {
    return (
      <div>
        <p className="text-xs text-ink-3">{data.label}</p>
        <div className="mt-1 flex items-center justify-between gap-3">
          <ClickToCopy className="mt-1" value={data.value}>
            <p
              className={`min-w-0 text-ink ${data.value.includes(" ") ? "break-all font-mono text-sm" : "font-mono text-lg"}`}
              dir="ltr"
            >
              {data.value}
            </p>
          </ClickToCopy>
        </div>
      </div>
    );
  }
  if (data?.kind === "value") {
    return (
      <div className="flex items-center gap-3">
        {data.swatch === "true" ? (
          <span className="size-10 shrink-0 rounded-xl border border-line" style={{ backgroundColor: data.value }} />
        ) : null}
        <ClickToCopy className="min-w-0 flex-1" value={data.value}>
          <p className="break-all font-mono text-sm text-ink" dir="ltr">
            {data.value}
          </p>
        </ClickToCopy>
      </div>
    );
  }
  return (
    <p className="text-sm leading-relaxed text-ink" dir="auto">
      {text}
      {answer.url ? (
        <a
          href={answer.url}
          {...newTabLinkProps(settings.results_on_new_tab)}
          className="ml-2 whitespace-nowrap text-xs text-accent hover:underline"
        >
          {hostname}
        </a>
      ) : null}
    </p>
  );
}
