// SPDX-License-Identifier: Apache-2.0 WITH Commons-Clause-1.0

import { LegacyAnswer } from "@/features/results/answers/Legacy.tsx";
import { TranslationsAnswer } from "@/features/results/answers/Translations.tsx";
import { WeatherAnswer } from "@/features/results/answers/Weather.tsx";
import { useT } from "@/lib/i18n.ts";
import type { AnswerData } from "@/lib/types.ts";

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
