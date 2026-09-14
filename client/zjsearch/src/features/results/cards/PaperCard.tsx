// SPDX-License-Identifier: Apache-2.0 WITH Commons-Clause-1.0

import { BookOpen, Calendar, ExternalLink, FileText, Fingerprint, Quote, User } from "lucide-react";
import { type ReactNode, useState } from "react";
import {
  type CardProps,
  EnginesLine,
  PrettyUrl,
  ResultArticle,
  ResultLink,
  Thumb,
  Title,
} from "@/features/results/cardParts.tsx";
import { formatDate } from "@/lib/format.ts";
import { useT } from "@/lib/i18n.ts";

/** Scholarly layout (science intent, arxiv/pubmed/...): authors · venue ·
    date meta line, clamped abstract, PDF/HTML actions and a compact DOI
    link.  Non-paper results on a science page (e.g. pdb figures) degrade
    gracefully - the card simply renders without the paper-specific bits. */

export function PaperCard({ result, globals }: CardProps) {
  const t = useT();
  const venueBits: string[] = [];
  // the resolver preference is a full URL ("https://oadoi.org/") - normalize
  // it instead of prepending a scheme (which produced https://https//...)
  const doiBase = globals.doi_resolver.replace(/\/+$/, "");
  const doiHref = doiBase.startsWith("http") ? `${doiBase}/${result.doi}` : `https://${doiBase}/${result.doi}`;
  if (result.journal) {
    venueBits.push(result.journal);
  }
  if (result.volume) {
    venueBits.push(`vol. ${result.volume}`);
  }
  if (result.number) {
    venueBits.push(`no. ${result.number}`);
  }
  if (result.pages) {
    venueBits.push(`pp. ${result.pages}`);
  }
  let authors = result.author ?? "";
  if (result.authors && result.authors.length > 0) {
    authors = result.authors.slice(0, 3).join(", ");
    if (result.authors.length > 3) {
      authors += ` ${t("et_al")}`;
    }
  }
  const metaBits: Array<string | null | ReactNode> = [
    result.published_date ? (
      <span className="inline-flex items-center gap-1" key="date">
        <Calendar className="size-3" />
        {formatDate(result.published_date)}
      </span>
    ) : null,
    authors ? (
      <span className="inline-flex items-center gap-1" key="authors">
        <User className="size-3 shrink-0" />
        {authors}
      </span>
    ) : null,
    venueBits.length > 0 ? (
      <span className="inline-flex items-center gap-1" key="venue">
        <BookOpen className="size-3 shrink-0" />
        {venueBits.join(", ")}
      </span>
    ) : null,
    result.comments ? (
      <span className="inline-flex items-center gap-1" key="citations">
        <Quote className="size-3 shrink-0" />
        {result.comments}
      </span>
    ) : null,
  ];
  // every optional extra (PDF/HTML, DOI, citation note, subject tags) lives
  // in ONE footer row in a fixed order, so cards share the same skeleton no
  // matter which fields the source engine provides
  const maxTags = 2;
  const tags = result.tags ?? [];
  const [tagsExpanded, setTagsExpanded] = useState(false);
  const hasFooter = Boolean(result.pdf_url || result.html_url || result.doi || result.comments || tags.length > 0);
  return (
    <ResultArticle priority={result.priority}>
      <div className="flex gap-4">
        <div className="min-w-0 flex-1">
          <PrettyUrl globals={globals} result={result} />
          <div className="mt-1">
            <Title globals={globals} result={result} />
          </div>
          {metaBits.some(Boolean) ? (
            // single line that swipes horizontally when it overflows (same
            // interaction as the filter rows) - touch users can reach the
            // truncated tail, no hover needed
            <p
              className="mt-1 flex items-center gap-1 overflow-x-auto whitespace-nowrap [scrollbar-width:none] [&::-webkit-scrollbar]:hidden [&>*]:shrink-0 text-xs text-ink-3"
              dir="auto"
            >
              {metaBits.filter(Boolean).map((bit, index) => (
                <span key={index}>
                  {index > 0 ? <span className="text-ink-3"> · </span> : null}
                  {bit}
                </span>
              ))}
            </p>
          ) : null}
          {result.content_html ? (
            <p
              className="mt-1.5 line-clamp-2 text-sm leading-relaxed text-ink-2"
              dangerouslySetInnerHTML={{ __html: result.content_html }}
              dir="auto"
            />
          ) : null}
          {hasFooter ? (
            // two stacked rows: actions + DOI, then topic tags as pills
            // with a +N fold - nothing truncates against its neighbours
            <div className="mt-2 space-y-1.5 text-xs">
              {result.pdf_url || result.html_url || result.doi ? (
                <div className="flex min-w-0 flex-wrap items-center gap-1">
                  {result.pdf_url ? (
                    <a
                      className="inline-flex shrink-0 items-center gap-1 rounded-full bg-accent-soft px-2 py-0.5 text-accent transition-colors hover:bg-accent-strong hover:text-accent-contrast"
                      href={result.pdf_url}
                      rel="noreferrer"
                      target="_blank"
                    >
                      <FileText className="size-3" />
                      PDF
                    </a>
                  ) : null}
                  {result.html_url ? (
                    <a
                      className="inline-flex shrink-0 items-center gap-1 rounded-full bg-surface-2 px-2 py-0.5 text-ink-2 hover:text-ink"
                      href={result.html_url}
                      rel="noreferrer"
                      target="_blank"
                    >
                      <ExternalLink className="size-3" />
                      HTML
                    </a>
                  ) : null}
                  {result.doi ? (
                    <a
                      className="inline-flex items-center gap-1 rounded-full bg-surface-2 px-2 py-0.5 font-mono text-[11px] leading-4 text-ink-2 hover:text-ink"
                      dir="ltr"
                      href={doiHref}
                      rel="noreferrer"
                      target="_blank"
                      title={`DOI ${result.doi}`}
                    >
                      <Fingerprint className="size-3 shrink-0" />
                      {result.doi}
                    </a>
                  ) : null}
                </div>
              ) : null}
              {tags.length > 0 ? (
                <div className="flex flex-wrap items-center gap-1">
                  {tags.slice(0, tagsExpanded ? tags.length : maxTags).map((tag) => (
                    <span className="rounded-full bg-surface-2 px-2 py-0.5 text-ink-3" key={tag} title={tag}>
                      #{tag}
                    </span>
                  ))}
                  {tags.length > maxTags ? (
                    <button
                      aria-expanded={tagsExpanded}
                      className="rounded-full bg-surface-2 px-2 py-0.5 text-ink-3 transition-colors hover:text-ink"
                      onClick={() => {
                        setTagsExpanded((value) => !value);
                      }}
                      title={tags.join(", ")}
                      type="button"
                    >
                      {tagsExpanded ? t("show_less") : `+${tags.length - maxTags}`}
                    </button>
                  ) : null}
                </div>
              ) : null}
            </div>
          ) : null}
        </div>
        {result.thumbnail ? (
          <ResultLink className="shrink-0 self-start" globals={globals} result={result}>
            <Thumb alt={result.title_text} className="h-28 w-28" src={result.thumbnail} />
          </ResultLink>
        ) : null}
      </div>
      <EnginesLine result={result} />
    </ResultArticle>
  );
}
