// SPDX-License-Identifier: AGPL-3.0-or-later

import { useState } from "react";
import type { CalculationAnswer } from "../../features/calculator.ts";
import { tryEvaluateExpression } from "../../features/calculator.ts";
import { useT } from "../../lib/i18n.ts";
import { useSettings } from "../../lib/settings.ts";
import type { AnswerData, WeatherItem } from "../../lib/types.ts";
import { LocationIcon } from "../icons.tsx";

const MAX_SOURCES_SHOWN = 3;

/** Interactive calculator for the "calculator" plugin: the query-detected
    expression seeds the display and the keypad keeps evaluating live, like
    the Google/DDG calculator cards. */
export function CalculatorAnswer({ calc }: { calc: CalculationAnswer }) {
  const [expression, setExpression] = useState(calc.expr);
  const result = tryEvaluateExpression(expression);
  const press = (key: string) => {
    setExpression((prev) => prev + key);
  };
  const keys: Array<{
    label: string;
    insert?: string;
    span?: string;
    kind?: "op" | "eq";
    action?: () => void;
  }> = [
    { label: "AC", action: () => setExpression("") },
    { label: "⌫", action: () => setExpression((prev) => prev.slice(0, -1)) },
    { label: "(", insert: "(" },
    { label: ")", insert: ")" },
    { label: "÷", insert: "/", kind: "op" },
    { label: "7", insert: "7" },
    { label: "8", insert: "8" },
    { label: "9", insert: "9" },
    { label: "×", insert: "*", kind: "op" },
    { label: "^", insert: "^", kind: "op" },
    { label: "4", insert: "4" },
    { label: "5", insert: "5" },
    { label: "6", insert: "6" },
    { label: "−", insert: "-", kind: "op" },
    { label: "%", insert: "%", kind: "op" },
    { label: "1", insert: "1" },
    { label: "2", insert: "2" },
    { label: "3", insert: "3" },
    { label: "+", insert: "+", kind: "op" },
    { label: "=", kind: "eq" },
    { label: "0", insert: "0", span: "col-span-3" },
    { label: ".", insert: "." },
  ];
  return (
    <div className="rounded-2xl border border-accent/25 bg-accent-soft/50 px-4 py-3 animate-fade-up">
      <div className="min-h-16 rounded-2xl border border-line bg-surface px-4 py-2 text-end">
        <p className="truncate text-xs text-ink-3" dir="ltr">
          {expression}
          {result ? " =" : ""}
        </p>
        <p className="min-h-10 truncate text-4xl font-semibold text-ink" dir="ltr">
          {result ? result.value : ""}
        </p>
      </div>
      <div className="mt-3 grid grid-cols-5 gap-2">
        {keys.map((key) => (
          <button
            className={`h-11 rounded-xl text-sm transition-colors ${
              key.kind === "eq"
                ? "row-span-2 bg-accent-strong text-base text-accent-contrast hover:opacity-90"
                : key.kind === "op"
                  ? "bg-surface-2 text-accent-strong hover:bg-line/40"
                  : "bg-surface-2 text-ink hover:bg-line/40"
            } ${key.span ?? ""}`}
            key={key.label}
            onClick={() => {
              if (key.action) {
                key.action();
              } else if (key.insert) {
                press(key.insert);
              }
            }}
            type="button"
          >
            {key.label}
          </button>
        ))}
      </div>
    </div>
  );
}

/** SVG temperature trend over the next hourly slots (accent area line with
    temp / time labels every third slot), horizontally scrollable. */
function WeatherTrend({ forecasts }: { forecasts: WeatherItem[] }) {
  const slots = forecasts.slice(0, 24);
  if (slots.length < 3) {
    return null;
  }
  const step = 38;
  const top = 22;
  const chartH = 46;
  const height = top + chartH + 22;
  const vals = slots.map((f) => f.temp_c);
  const min = Math.min(...vals);
  const max = Math.max(...vals);
  const span = max - min || 1;
  const points = slots.map((f, i) => ({
    x: i * step + step / 2,
    y: top + chartH * (1 - (f.temp_c - min) / span),
  }));
  const line = points.map((p, i) => `${i === 0 ? "M" : "L"}${p.x.toFixed(1)} ${p.y.toFixed(1)}`).join(" ");
  const endX = (points.length - 1) * step + step / 2;
  const area = `${line} L${endX.toFixed(1)} ${top + chartH} L${(step / 2).toFixed(1)} ${top + chartH} Z`;
  const labels: Array<{ x: number; y: number; temp: number; time: string }> = [];
  points.forEach((p, i) => {
    const slot = slots[i];
    if (slot && i % 3 === 0) {
      labels.push({ x: p.x, y: p.y, temp: slot.temp_c, time: slot.time ?? "" });
    }
  });
  return (
    <div className="mt-4 overflow-x-auto pb-1 [scrollbar-width:none] [&::-webkit-scrollbar]:hidden">
      <svg className="block" height={height} role="img" width={slots.length * step}>
        <path className="fill-accent/15" d={area} />
        <path className="stroke-accent-strong" d={line} fill="none" strokeLinecap="round" strokeWidth={2} />
        {labels.map((label) => (
          <g key={label.x}>
            <text className="fill-ink text-[10px] font-semibold" textAnchor="middle" x={label.x} y={label.y - 7}>
              {Math.round(label.temp)}°
            </text>
            <text className="fill-ink-3 text-[10px]" textAnchor="middle" x={label.x} y={top + chartH + 14}>
              {label.time}
            </text>
          </g>
        ))}
      </svg>
    </div>
  );
}

/** Daily strip grouped from the hourly slots: weekday, mid-day symbol and
    the day's high/low temperature (Google weather style). */
function WeatherDaily({ forecasts }: { forecasts: WeatherItem[] }) {
  const days: Array<{
    date: string;
    weekday: string;
    symbol: string;
    hi: number;
    lo: number;
    bestHour: number;
  }> = [];
  const byDate = new Map<string, (typeof days)[number]>();
  for (const f of forecasts) {
    if (!f.date_iso) {
      continue;
    }
    let day = byDate.get(f.date_iso);
    if (!day) {
      day = {
        date: f.date_iso,
        weekday: f.weekday ?? f.date_iso,
        symbol: f.symbol,
        hi: f.temp_c,
        lo: f.temp_c,
        bestHour: f.hour ?? 12,
      };
      byDate.set(f.date_iso, day);
      days.push(day);
      continue;
    }
    if (f.temp_c > day.hi) {
      day.hi = f.temp_c;
    }
    if (f.temp_c < day.lo) {
      day.lo = f.temp_c;
    }
    const hour = f.hour ?? 12;
    if (f.symbol && Math.abs(hour - 13) < Math.abs(day.bestHour - 13)) {
      day.bestHour = hour;
      day.symbol = f.symbol;
    }
  }
  if (days.length < 2) {
    return null;
  }
  return (
    <div className="mt-4 flex gap-2 overflow-x-auto pb-1">
      {days.map((day, index) => (
        <div
          className={`w-20 shrink-0 rounded-2xl border px-2 py-2.5 text-center ${
            index === 0 ? "border-accent/40 bg-accent-soft/40" : "border-line bg-surface"
          }`}
          key={day.date}
        >
          <p className="text-xs font-medium text-ink">{day.weekday}</p>
          {day.symbol ? <img alt="" className="mx-auto mt-1.5 size-8" src={day.symbol} /> : null}
          <p className="mt-1.5 whitespace-nowrap text-xs">
            <span className="font-semibold text-ink">{Math.round(day.hi)}°</span>{" "}
            <span className="text-ink-3">{Math.round(day.lo)}°</span>
          </p>
        </div>
      ))}
    </div>
  );
}

function WeatherAnswer({
  answer,
  sources,
}: {
  answer: Extract<AnswerData, { template: "answer/weather.html" }>;
  sources: Array<{ service: string; url: string }>;
}) {
  const t = useT();
  // same cap-and-expand contract as EnginesLine: 3 pills + "+N"
  const [sourcesExpanded, setSourcesExpanded] = useState(false);
  const shownSources = sourcesExpanded ? sources : sources.slice(0, MAX_SOURCES_SHOWN);
  const hiddenSources = sources.length - MAX_SOURCES_SHOWN;
  const current = answer.current;
  const heroC = Math.round(current.temp_c);
  const heroF = Math.round(current.temp_f);
  const meta: Array<[string, string]> = [];
  if (current.feels_like) {
    meta.push([t("feels_like"), current.feels_like]);
  }
  if (current.wind) {
    meta.push([t("wind"), current.wind_speed ? `${current.wind} ${current.wind_speed}` : current.wind]);
  }
  if (current.humidity) {
    meta.push([t("humidity"), current.humidity]);
  }
  if (current.pressure) {
    meta.push([t("pressure"), current.pressure]);
  }
  return (
    <div>
      <p className="flex items-center gap-1.5 text-sm font-medium text-ink">
        <LocationIcon className="size-4 shrink-0 text-ink-3" />
        {current.location_name}
      </p>
      <div className="mt-3 flex flex-wrap items-center justify-between gap-x-10 gap-y-3">
        <div className="flex items-center gap-4">
          {current.symbol ? <img alt="" className="size-16" src={current.symbol} /> : null}
          <div>
            <p className="flex items-start gap-2">
              <span className="text-5xl font-semibold leading-none text-ink">
                {heroC}
                <span className="ms-0.5 align-top text-lg font-medium text-ink-3">°C</span>
              </span>
              <span className="mt-1 border-s border-line ps-2 text-sm text-ink-3">{heroF} °F</span>
            </p>
            <p className="mt-2 text-sm text-ink-2">{current.condition_display}</p>
          </div>
        </div>
        <dl className="grid grid-cols-2 gap-x-10 gap-y-1 text-xs">
          {meta.map(([label, value]) => (
            <div key={label}>
              <span className="text-ink-3">{label}: </span>
              <span className="font-medium text-ink">{value}</span>
            </div>
          ))}
        </dl>
      </div>
      <WeatherTrend forecasts={answer.forecasts} />
      <WeatherDaily forecasts={answer.forecasts} />
      {sources.length > 0 ? (
        <div className="mt-3 flex min-w-0 flex-wrap items-center gap-x-2 gap-y-1 text-xs text-ink-3">
          {shownSources.map((source) =>
            source.url ? (
              <a
                className="inline-flex items-center rounded-full bg-surface-2 px-2 py-0.5 text-ink-2 transition-colors hover:text-ink"
                href={source.url}
                key={source.service}
                rel="noreferrer"
                target="_blank"
              >
                {source.service}
              </a>
            ) : (
              <span className="rounded-full bg-surface-2 px-2 py-0.5" key={source.service}>
                {source.service}
              </span>
            ),
          )}
          {!sourcesExpanded && hiddenSources > 0 ? (
            <button
              className="rounded-full bg-surface-2 px-2 py-0.5 transition-colors hover:text-ink"
              onClick={() => {
                setSourcesExpanded(true);
              }}
              title={sources.map((source) => source.service).join(", ")}
              type="button"
            >
              +{hiddenSources}
            </button>
          ) : null}
          {sourcesExpanded && hiddenSources > 0 ? (
            <button
              className="transition-colors hover:text-ink"
              onClick={() => {
                setSourcesExpanded(false);
              }}
              type="button"
            >
              {t("show_less")}
            </button>
          ) : null}
        </div>
      ) : null}
    </div>
  );
}

/** Translation engines stamp the language pair into their answer URLs in
    engine-specific shapes: lingva uses /from/to/ path segments, mymemory
    uses sl/tl query parameters, libretranslate uses source/target. */
function parseLangPair(url: string): { from: string; to: string } | null {
  let parsed: URL;
  try {
    parsed = new URL(url);
  } catch {
    return null;
  }
  const lang = /^[a-z]{2,3}(?:-[a-zA-Z]{2,4})?$/i;
  const [pathFrom, pathTo] = parsed.pathname.split("/").filter(Boolean);
  if (pathFrom && pathTo && lang.test(pathFrom) && lang.test(pathTo)) {
    return { from: pathFrom, to: pathTo };
  }
  const from = parsed.searchParams.get("source") ?? parsed.searchParams.get("sl") ?? parsed.searchParams.get("from");
  const to = parsed.searchParams.get("target") ?? parsed.searchParams.get("tl") ?? parsed.searchParams.get("to");
  if (from && to && lang.test(from) && lang.test(to)) {
    return { from, to };
  }
  return null;
}

/** Dictionary/translation answer.  Two layouts by payload shape:
    - translation (lingva, no definitions): language-pair chip + the translated
      text as the hero;
    - dictionary (wordnik): the queried word heads numbered definitions — the
      definition IS the answer, so nothing important hides behind a collapse. */
function TranslationsAnswer({
  answer,
  query,
}: {
  answer: Extract<AnswerData, { template: "answer/translations.html" }>;
  query?: string;
}) {
  const t = useT();
  const first = answer.translations[0];
  if (!first) {
    return null;
  }
  const rest = answer.translations.slice(1);
  // the raw query keeps its bang tokens and the "en-de " language-pair prefix
  // of dictionary engines — both are noise everywhere it could be displayed
  const cleanQuery = query
    ?.replace(/^(\s*![^\s]+)+/, "")
    .replace(/^\s*[a-z]{2,3}-[a-zA-Z]{2,4}\s+/i, "")
    .trim();
  const langPair = parseLangPair(answer.url);
  const showExamples = (max: number) =>
    first.examples.length > 0 ? (
      <div className="mt-2 space-y-1">
        {first.examples.slice(0, max).map((example, i) => (
          <p className="text-sm italic leading-relaxed text-ink-2" dir="auto" key={i}>
            “{example}”
          </p>
        ))}
      </div>
    ) : null;
  if (first.definitions.length === 0) {
    // pure translation (lingva, mymemory): the translated text is the hero,
    // alternatives and match examples follow
    return (
      <div>
        <div className="flex flex-wrap items-center gap-2">
          {langPair ? (
            <span className="rounded-full bg-surface-2 px-2 py-0.5 font-mono text-xs text-ink-2" dir="ltr">
              {langPair.from} → {langPair.to}
            </span>
          ) : null}
          {cleanQuery && cleanQuery !== first.text ? (
            <span className="min-w-0 truncate text-xs text-ink-3" dir="auto">
              {cleanQuery}
            </span>
          ) : null}
        </div>
        <p className="mt-2.5 text-xl font-medium leading-snug text-ink" dir="auto">
          {first.text}
        </p>
        {showExamples(3)}
        {rest.length > 0 ? (
          <div className="mt-3 space-y-2 border-t border-line pt-3">
            {rest.map((item, index) => (
              <div key={index}>
                <p className="text-sm font-medium text-ink" dir="auto">
                  {item.text}
                </p>
                {item.definitions.map((definition, i) => (
                  <p className="mt-0.5 text-xs leading-relaxed text-ink-2" dir="auto" key={i}>
                    {definition}
                  </p>
                ))}
              </div>
            ))}
          </div>
        ) : null}
        <div className="mt-3 flex min-w-0 flex-wrap items-center gap-x-2 gap-y-1 text-xs text-ink-3">
          <span className="rounded-full bg-surface-2 px-2 py-0.5">{answer.engine}</span>
        </div>
      </div>
    );
  }
  const hasMore =
    first.definitions.length > 4 || rest.length > 0 || first.examples.length > 0 || first.synonyms.length > 0;
  return (
    <div>
      <p className="text-lg font-semibold text-ink" dir="auto">
        {cleanQuery ?? first.text}
        {first.transliteration ? (
          <span className="ms-2 text-sm font-normal text-ink-3">{first.transliteration}</span>
        ) : null}
      </p>
      {first.definitions.length > 0 ? (
        <ol className="mt-2 space-y-1.5">
          {first.definitions.slice(0, 4).map((definition, i) => (
            <li className="flex gap-2 text-sm leading-relaxed text-ink-2" key={i}>
              <span className="shrink-0 text-ink-3">{i + 1}.</span>
              <span dir="auto">{definition}</span>
            </li>
          ))}
        </ol>
      ) : null}
      {showExamples(2)}
      {first.examples.length > 0 ? (
        <div className="mt-2 space-y-1">
          {first.examples.slice(0, 2).map((example, i) => (
            <p className="text-sm italic leading-relaxed text-ink-2" dir="auto" key={i}>
              “{example}”
            </p>
          ))}
        </div>
      ) : null}
      {first.synonyms.length > 0 ? (
        <div className="mt-2 flex flex-wrap items-center gap-x-2 gap-y-1 text-xs text-ink-3">
          <span>{t("synonyms")}:</span>
          {first.synonyms.slice(0, 6).map((synonym) => (
            <span className="rounded-full bg-surface-2 px-2 py-0.5 text-ink-2" dir="auto" key={synonym}>
              {synonym}
            </span>
          ))}
        </div>
      ) : null}
      {hasMore ? (
        <details className="mt-2">
          <summary className="cursor-pointer text-xs text-ink-3 transition-colors hover:text-ink">
            {t("definitions")}
          </summary>
          <div className="mt-2 space-y-3">
            {first.definitions.length > 4 ? (
              <ol className="space-y-1.5" start={5}>
                {first.definitions.slice(4).map((definition, i) => (
                  <li className="flex gap-2 text-sm leading-relaxed text-ink-2" key={i}>
                    <span className="shrink-0 text-ink-3">{i + 5}.</span>
                    <span dir="auto">{definition}</span>
                  </li>
                ))}
              </ol>
            ) : null}
            {rest.map((item, index) => (
              <div className="text-sm" key={index}>
                <p className="font-medium text-ink" dir="auto">
                  {item.text}
                  {item.transliteration ? (
                    <span className="ms-1.5 font-normal text-ink-3">{item.transliteration}</span>
                  ) : null}
                </p>
                {item.definitions.length > 0 ? (
                  <ol className="mt-1 space-y-1.5">
                    {item.definitions.map((definition, i) => (
                      <li className="flex gap-2 text-ink-2" key={i}>
                        <span className="shrink-0 text-ink-3">{i + 1}.</span>
                        <span dir="auto">{definition}</span>
                      </li>
                    ))}
                  </ol>
                ) : null}
              </div>
            ))}
            {first.synonyms.length > 0 ? (
              <div className="flex flex-wrap items-center gap-x-2 gap-y-1 text-xs text-ink-3">
                <span>{t("synonyms")}:</span>
                {first.synonyms.slice(6).map((synonym) => (
                  <span className="rounded-full bg-surface-2 px-2 py-0.5 text-ink-2" dir="auto" key={synonym}>
                    {synonym}
                  </span>
                ))}
              </div>
            ) : null}
          </div>
        </details>
      ) : null}
      {answer.engine ? <p className="mt-2 text-xs text-ink-3">{answer.engine}</p> : null}
    </div>
  );
}

function CopyButton({ value }: { value: string }) {
  const t = useT();
  const [copied, setCopied] = useState(false);
  return (
    <button
      className="shrink-0 rounded-lg border border-line px-2 py-1 text-xs text-ink-3 transition-colors hover:text-ink"
      onClick={() => {
        navigator.clipboard?.writeText(value).then(() => {
          setCopied(true);
          window.setTimeout(() => setCopied(false), 1200);
        });
      }}
      type="button"
    >
      {copied ? t("copied") : t("copy")}
    </button>
  );
}

/** Special-query answers (random, statistics, hash, self-info, time zone)
    arrive as plain legacy text; the patterns below give each of them a
    purpose-built layout, falling back to plain text for anything else. */
function LegacyAnswer({ answer }: { answer: Extract<AnswerData, { template: "answer/legacy.html" }> }) {
  const settings = useSettings();
  const text = answer.answer;
  const hashMatch = /^(.+?)\s*(?:hash digest|散列摘要)\s*:\s*([a-f0-9]{32,128})$/i.exec(text);
  const statsMatch = /^\[(.+?)\] (\w+)\((.+)\) = (.+?)\s*$/.exec(text);
  const zoneMatch = /^(.+?): (.+ \d[^)]*) \(([A-Z]{2,5})\)$/.exec(text);
  const ipMatch = /^(.*IP.*?[：:])\s*(\d{1,3}(?:\.\d{1,3}){3})$/u.exec(text);
  const uaMatch = /^(.*(?:user-agent|用户代理).*?[：:])\s*(.+)$/iu.exec(text);
  const isColor = /^#[0-9a-f]{6}$/i.test(text);
  const isUuid = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i.test(text);
  const isBareValue = !isColor && !isUuid && !/\s/.test(text) && text.length <= 64;

  if (hashMatch) {
    const algo = hashMatch[1] ?? "";
    const digest = hashMatch[2] ?? "";
    return (
      <div>
        <div className="flex items-center justify-between gap-3">
          <span className="rounded-full bg-surface-2 px-2 py-0.5 font-mono text-xs text-ink-2" dir="ltr">
            {algo}
          </span>
          <CopyButton value={digest} />
        </div>
        <p className="mt-2 break-all font-mono text-sm text-ink" dir="ltr">
          {digest}
        </p>
      </div>
    );
  }
  if (statsMatch) {
    const fn = statsMatch[2] ?? "";
    const args = statsMatch[3] ?? "";
    const result = statsMatch[4] ?? "";
    return (
      <div>
        <p className="truncate text-xs text-ink-3" dir="ltr">
          <span className="font-mono font-medium text-accent-strong">{fn}</span>({args})
        </p>
        <div className="mt-1 flex items-center justify-between gap-3">
          <p className="text-2xl font-semibold text-ink" dir="ltr">
            {result}
          </p>
          <CopyButton value={result} />
        </div>
      </div>
    );
  }
  if (zoneMatch) {
    const zone = zoneMatch[1] ?? "";
    const time = zoneMatch[2] ?? "";
    const abbr = zoneMatch[3] ?? "";
    return (
      <div className="flex items-center justify-between gap-3">
        <div className="min-w-0">
          <p className="truncate font-mono text-xs text-ink-3" dir="ltr">
            {zone}
          </p>
          <p className="mt-1 text-xl font-medium text-ink" dir="auto">
            {time}
          </p>
        </div>
        <span className="shrink-0 rounded-full bg-surface-2 px-2 py-0.5 font-mono text-xs text-ink-2">{abbr}</span>
      </div>
    );
  }
  if (ipMatch || uaMatch) {
    const match = ipMatch ?? uaMatch;
    const label = (match?.[1] ?? "").replace(/[：:]\s*$/, "");
    const value = match?.[2] ?? "";
    return (
      <div>
        <p className="text-xs text-ink-3">{label}</p>
        <div className="mt-1 flex items-center justify-between gap-3">
          <p className={`min-w-0 text-ink ${ipMatch ? "font-mono text-lg" : "break-all font-mono text-sm"}`} dir="ltr">
            {value}
          </p>
          <CopyButton value={value} />
        </div>
      </div>
    );
  }
  if (isColor || isUuid || isBareValue) {
    return (
      <div className="flex items-center gap-3">
        {isColor ? (
          <span className="size-10 shrink-0 rounded-xl border border-line" style={{ backgroundColor: text }} />
        ) : null}
        <p className="min-w-0 flex-1 break-all font-mono text-sm text-ink" dir="ltr">
          {text}
        </p>
        <CopyButton value={text} />
      </div>
    );
  }
  let hostname = "";
  if (answer.url) {
    try {
      hostname = new URL(answer.url).hostname;
    } catch {
      hostname = answer.url;
    }
  }
  return (
    <p className="text-sm leading-relaxed text-ink" dir="auto">
      {text}
      {answer.url ? (
        <a
          href={answer.url}
          {...(settings.results_on_new_tab ? { target: "_blank", rel: "noopener noreferrer" } : { rel: "noreferrer" })}
          className="ml-2 whitespace-nowrap text-xs text-accent hover:underline"
        >
          {hostname}
        </a>
      ) : null}
    </p>
  );
}

export function Answers({ answers, query }: { answers: AnswerData[]; query?: string }) {
  const t = useT();
  if (answers.length === 0) {
    return null;
  }
  // several weather engines may answer the same query with the same data at
  // different coverage (duckduckgo ~10 days hourly, open-meteo ~2.7 days,
  // wttr.in 3 days 3-hourly): render the longest coverage as one card and
  // credit every answering engine as its source
  const weatherAnswers = answers.filter(
    (answer): answer is Extract<AnswerData, { template: "answer/weather.html" }> =>
      answer.template === "answer/weather.html",
  );
  weatherAnswers.sort((a, b) => b.forecasts.length - a.forecasts.length);
  const weatherSources = weatherAnswers.map((a) => ({ service: a.service, url: a.url }));
  const longest = weatherAnswers[0];
  const visible: AnswerData[] = [];
  let weatherInserted = false;
  for (const answer of answers) {
    if (answer.template === "answer/weather.html") {
      if (!weatherInserted && longest !== undefined) {
        visible.push(longest);
        weatherInserted = true;
      }
      continue;
    }
    visible.push(answer);
  }
  return (
    <section aria-label={t("answers")} className="space-y-2">
      {visible.map((answer, index) => (
        <div className="rounded-2xl border border-accent/25 bg-accent-soft/50 px-4 py-3 animate-fade-up" key={index}>
          {answer.template === "answer/translations.html" ? (
            <TranslationsAnswer answer={answer} query={query} />
          ) : answer.template === "answer/weather.html" ? (
            <WeatherAnswer answer={answer} sources={weatherSources} />
          ) : (
            <LegacyAnswer answer={answer} />
          )}
        </div>
      ))}
    </section>
  );
}
