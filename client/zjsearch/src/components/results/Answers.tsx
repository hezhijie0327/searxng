// SPDX-License-Identifier: AGPL-3.0-or-later

import { useT } from "../../lib/i18n.ts";
import { useSettings } from "../../lib/settings.ts";
import type { AnswerData, WeatherItem } from "../../lib/types.ts";
import { LocationIcon } from "../icons.tsx";

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

function WeatherAnswer({ answer }: { answer: Extract<AnswerData, { template: "answer/weather.html" }> }) {
  const t = useT();
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
      <div className="flex items-center justify-between gap-3">
        <p className="flex items-center gap-1.5 text-sm font-medium text-ink">
          <LocationIcon className="size-4 shrink-0 text-ink-3" />
          {current.location_name}
        </p>
        {answer.service ? (
          answer.url ? (
            <a
              className="text-xs text-ink-3 transition-colors hover:text-ink hover:underline"
              href={answer.url}
              rel="noreferrer"
              target="_blank"
            >
              {answer.service}
            </a>
          ) : (
            <p className="text-xs text-ink-3">{answer.service}</p>
          )
        ) : null}
      </div>
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
    </div>
  );
}

function TranslationsAnswer({ answer }: { answer: Extract<AnswerData, { template: "answer/translations.html" }> }) {
  const t = useT();
  const first = answer.translations[0];
  if (!first) {
    return null;
  }
  return (
    <div>
      <p className="text-base font-medium text-ink" dir="auto">
        {first.text}
        {first.transliteration ? (
          <span className="ml-2 text-sm font-normal text-ink-3">{first.transliteration}</span>
        ) : null}
      </p>
      {answer.translations.length > 1 ||
      first.definitions.length > 0 ||
      first.examples.length > 0 ||
      first.synonyms.length > 0 ? (
        <details className="mt-1.5">
          <summary className="cursor-pointer text-xs text-ink-3 transition-colors hover:text-ink">
            {t("definitions")}
          </summary>
          <div className="mt-2 space-y-3">
            {answer.translations.map((item, index) => (
              <div className="text-xs" key={index}>
                <p className="font-medium text-ink" dir="auto">
                  {item.text}
                  {item.transliteration ? (
                    <span className="ml-1.5 font-normal text-ink-3">{item.transliteration}</span>
                  ) : null}
                </p>
                {item.definitions.length > 0 ? (
                  <div className="mt-1">
                    <span className="text-ink-3">{t("definitions")}:</span>
                    <ul className="ml-4 list-disc">
                      {item.definitions.map((definition, i) => (
                        <li className="text-ink-2" key={i}>
                          {definition}
                        </li>
                      ))}
                    </ul>
                  </div>
                ) : null}
                {item.examples.length > 0 ? (
                  <div className="mt-1">
                    <span className="text-ink-3">{t("examples")}:</span>
                    <ul className="ml-4 list-disc">
                      {item.examples.map((example, i) => (
                        <li className="text-ink-2" key={i}>
                          {example}
                        </li>
                      ))}
                    </ul>
                  </div>
                ) : null}
                {item.synonyms.length > 0 ? (
                  <div className="mt-1">
                    <span className="text-ink-3">{t("synonyms")}:</span>
                    <span className="ml-1 text-ink-2">{item.synonyms.join(", ")}</span>
                  </div>
                ) : null}
              </div>
            ))}
          </div>
        </details>
      ) : null}
      {answer.engine ? <p className="mt-1.5 text-xs text-ink-3">{answer.engine}</p> : null}
    </div>
  );
}

function LegacyAnswer({ answer }: { answer: Extract<AnswerData, { template: "answer/legacy.html" }> }) {
  const settings = useSettings();
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
      {answer.answer}
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

export function Answers({ answers }: { answers: AnswerData[] }) {
  const t = useT();
  if (answers.length === 0) {
    return null;
  }
  // several weather engines may answer the same query; one big card is the
  // whole point of the weather presentation, so keep the first only
  let weatherSeen = false;
  const visible = answers.filter((answer) => {
    if (answer.template !== "answer/weather.html") {
      return true;
    }
    if (weatherSeen) {
      return false;
    }
    weatherSeen = true;
    return true;
  });
  return (
    <section aria-label={t("answers")} className="space-y-2">
      {visible.map((answer, index) => (
        <div className="rounded-2xl border border-accent/25 bg-accent-soft/50 px-4 py-3 animate-fade-up" key={index}>
          {answer.template === "answer/translations.html" ? (
            <TranslationsAnswer answer={answer} />
          ) : answer.template === "answer/weather.html" ? (
            <WeatherAnswer answer={answer} />
          ) : (
            <LegacyAnswer answer={answer} />
          )}
        </div>
      ))}
    </section>
  );
}
