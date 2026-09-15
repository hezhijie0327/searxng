// SPDX-License-Identifier: Apache-2.0 WITH Commons-Clause-1.0

import { LegacyAnswer } from "@/features/results/answers/Legacy.tsx";
import { TranslationsAnswer } from "@/features/results/answers/Translations.tsx";
import { WeatherAnswer } from "@/features/results/answers/Weather.tsx";
import { useT } from "@/lib/i18n.ts";
import type { AnswerData } from "@/lib/types.ts";

/** Answer tiering (see AGENTS.md): rich widgets (weather, translations) and
    the interactive converter keep the accent card; every other legacy answer
    (calculator, time, ip, hash, random, stats, tor — none has a source url)
    renders uncarded with a border-b divider, Google-style. */
function isCarded(answer: AnswerData): boolean {
  return answer.template !== "answer/legacy.html" || answer.data?.kind === "unit_conversion";
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
      {visible.map((answer, index) => {
        const body =
          answer.template === "answer/translations.html" ? (
            <TranslationsAnswer answer={answer} key={index} query={query} />
          ) : answer.template === "answer/weather.html" ? (
            <WeatherAnswer answer={answer} key={index} sources={weatherSources} />
          ) : (
            <LegacyAnswer answer={answer} key={index} />
          );
        return isCarded(answer) ? (
          <div className="animate-fade-up rounded-2xl border border-accent/25 bg-accent-soft/50 px-4 py-3" key={index}>
            {body}
          </div>
        ) : (
          <div className="animate-fade-up border-b border-line px-4 pb-3 last:border-b-0" key={index}>
            {body}
          </div>
        );
      })}
    </section>
  );
}
