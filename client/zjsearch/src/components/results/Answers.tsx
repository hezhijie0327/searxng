// SPDX-License-Identifier: AGPL-3.0-or-later

import { useT } from "../../lib/i18n.ts";
import { useSettings } from "../../lib/settings.ts";
import type { AnswerData, WeatherItem } from "../../lib/types.ts";
import { ClockIcon } from "../icons.tsx";

function WeatherGrid({ item }: { item: WeatherItem }) {
  const t = useT();
  const cells: Array<[string, string]> = [[t("temperature"), item.temperature]];
  if (item.feels_like) {
    cells.push([t("feels_like"), item.feels_like]);
  }
  if (item.wind) {
    cells.push([t("wind"), item.wind_speed ? `${item.wind}: ${item.wind_speed}` : item.wind]);
  }
  if (item.pressure) {
    cells.push([t("pressure"), item.pressure]);
  }
  if (item.humidity) {
    cells.push([t("humidity"), item.humidity]);
  }
  return (
    <div className="grid grid-cols-2 gap-x-6 gap-y-1 text-xs sm:grid-cols-4">
      {cells.map(([label, value]) => (
        <div key={label}>
          <span className="text-ink-3">{label}: </span>
          <span className="font-medium text-ink">{value}</span>
        </div>
      ))}
    </div>
  );
}

function WeatherAnswer({ answer }: { answer: Extract<AnswerData, { template: "answer/weather.html" }> }) {
  const current = answer.current;
  return (
    <div>
      <div className="flex items-start gap-3">
        {current.symbol ? (
          <img alt="" className="size-10" decoding="async" loading="lazy" src={current.symbol} />
        ) : null}
        <div className="min-w-0">
          <p className="text-sm font-medium text-ink" dir="auto">
            {current.summary}
          </p>
          <div className="mt-1.5">
            <WeatherGrid item={current} />
          </div>
        </div>
      </div>
      {answer.forecasts.length > 0 ? (
        <details className="mt-2">
          <summary className="cursor-pointer text-xs text-ink-3 transition-colors hover:text-ink">
            {answer.forecasts.length > 0 ? "Forecast" : ""}
          </summary>
          <div className="mt-2 space-y-2 border-l border-line pl-3">
            {answer.forecasts.map((forecast, index) => (
              <div key={index}>
                <p className="text-xs text-ink-2" dir="auto">
                  <span className="inline-flex items-center gap-1 font-medium">
                    <ClockIcon className="size-3" />
                    {forecast.datetime_display}
                  </span>{" "}
                  — {forecast.summary}
                </p>
                <div className="mt-1">
                  <WeatherGrid item={forecast} />
                </div>
              </div>
            ))}
          </div>
        </details>
      ) : null}
      {answer.service ? <p className="mt-2 text-xs text-ink-3">{answer.service}</p> : null}
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
  return (
    <section aria-label={t("answers")} className="space-y-2">
      {answers.map((answer, index) => (
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
