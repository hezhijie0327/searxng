// SPDX-License-Identifier: Apache-2.0 WITH Commons-Clause-1.0

import { type CardProps, EnginesLine, ResultArticle, ResultLink } from "@/features/results/cardParts.tsx";

/** Dictionary entry card for the dictionaries/define categories: the headword
    is the identity (no URL chrome), a phonetic chip is lifted out of the
    Wiktionary blob ("IPA(key): /…/" is stable Wiktionary markup). */

export function DictionaryCard({ result, globals }: CardProps) {
  const ipa = /IPA\(key\):\s*\/([^/]+)\//.exec(result.content_html ?? "");
  const content = ipa ? (result.content_html ?? "").replaceAll(ipa[0], "") : result.content_html;
  return (
    <ResultArticle priority={result.priority}>
      <div className="mt-1 flex flex-wrap items-baseline gap-x-3 gap-y-1">
        <h3 className="line-clamp-2 min-w-0 text-base font-medium leading-snug">
          <ResultLink
            className="text-ink decoration-accent/50 underline-offset-2 hover:text-accent hover:underline"
            globals={globals}
            result={result}
          >
            <span dangerouslySetInnerHTML={{ __html: result.title_html }} dir="auto" />
          </ResultLink>
        </h3>
        {ipa ? (
          <code className="rounded bg-surface-2 px-1.5 py-0.5 font-mono text-xs text-ink-2" dir="ltr">
            /{ipa[1]}/
          </code>
        ) : null}
      </div>
      {content ? (
        <p
          className="mt-1.5 line-clamp-3 text-sm leading-relaxed text-ink-2"
          dangerouslySetInnerHTML={{ __html: content }}
          dir="auto"
        />
      ) : null}
      <EnginesLine result={result} />
    </ResultArticle>
  );
}
