// SPDX-License-Identifier: Apache-2.0 WITH Commons-Clause-1.0

import { Calendar, ChevronLeft, Code, ExternalLink, Scale, Star, User } from "lucide-react";
import { type ReactNode, useState } from "react";
import { type CardProps, EnginesLine, PrettyUrl, ResultArticle, Title } from "@/features/results/cardParts.tsx";
import { formatDate } from "@/lib/format.ts";
import { useT } from "@/lib/i18n.ts";

export function PackageCard({ result, globals }: CardProps) {
  const t = useT();
  const [tagsExpanded, setTagsExpanded] = useState(false);
  // same slot rhythm as DefaultCard: url / title / meta / content / engines;
  // secondary links fold into the engines row so no card grows extra rows
  const links: Array<{ icon: ReactNode; label: string; url: string }> = [];
  if (result.homepage) {
    links.push({ icon: <ExternalLink className="size-3" />, label: "Homepage", url: result.homepage });
  }
  if (result.source_code_url && result.source_code_url !== result.url) {
    links.push({ icon: <Code className="size-3" />, label: "Source code", url: result.source_code_url });
  }
  for (const [name, url] of Object.entries(result.project_links ?? {})) {
    if (url !== result.url) {
      links.push({ icon: <ExternalLink className="size-3" />, label: name, url });
    }
  }
  return (
    <ResultArticle priority={result.priority}>
      <PrettyUrl globals={globals} result={result} />
      <div className="mt-1">
        <Title globals={globals} result={result} />
      </div>
      {result.published_date || result.maintainer || result.popularity || result.license_name || result.version ? (
        <div className="mt-1 flex items-center gap-x-3 overflow-x-auto whitespace-nowrap [scrollbar-width:none] [&::-webkit-scrollbar]:hidden [&>*]:shrink-0 text-xs text-ink-3">
          {result.published_date ? (
            <span className="inline-flex items-center gap-1" key="date">
              <Calendar className="size-3" />
              {formatDate(result.published_date)}
            </span>
          ) : null}
          {result.maintainer ? (
            <span className="inline-flex min-w-0 items-center gap-1" key="author">
              <User className="size-3 shrink-0" />
              {t("author")}:
              <span className="truncate text-ink-2" dir="auto">
                {result.maintainer}
              </span>
            </span>
          ) : null}
          {result.popularity ? (
            <span className="inline-flex items-center gap-1" key="popularity">
              <Star className="size-3" />
              <span className="text-ink-2">{result.popularity}</span>
            </span>
          ) : null}
          {result.license_name ? (
            <span className="inline-flex items-center gap-1" key="license">
              <Scale className="size-3 shrink-0" />
              {t("license")}:
              <span className="text-ink-2">
                {result.license_url ? (
                  <a
                    className="hover:text-accent hover:underline"
                    href={result.license_url}
                    rel="noreferrer"
                    target="_blank"
                  >
                    {result.license_name}
                  </a>
                ) : (
                  result.license_name
                )}
              </span>
            </span>
          ) : null}
          {result.version ? <span key="version">v{result.version}</span> : null}
        </div>
      ) : null}
      {result.content_html ? (
        <p
          className="mt-1.5 line-clamp-2 text-sm leading-relaxed text-ink-2"
          dangerouslySetInnerHTML={{ __html: result.content_html }}
          dir="auto"
        />
      ) : null}
      {result.tags && result.tags.length > 0 ? (
        <div className="mt-1.5 flex flex-wrap items-center gap-1 text-xs">
          {result.tags.slice(0, tagsExpanded ? result.tags.length : 4).map((tag) => (
            <span className="rounded-full bg-surface-2 px-2 py-0.5 text-ink-3" key={tag} title={tag}>
              #{tag}
            </span>
          ))}
          {result.tags.length > 4 ? (
            <button
              aria-expanded={tagsExpanded}
              className="inline-flex items-center gap-1 rounded-full bg-surface-2 px-2 py-0.5 text-ink-3 transition-colors hover:text-ink"
              onClick={() => {
                setTagsExpanded((value) => !value);
              }}
              title={result.tags.join(", ")}
              type="button"
            >
              {tagsExpanded ? (
                <>
                  <ChevronLeft className="size-3 shrink-0" />
                  {t("show_less")}
                </>
              ) : (
                `+${result.tags.length - 4}`
              )}
            </button>
          ) : null}
        </div>
      ) : null}
      <EnginesLine
        leading={
          links.length > 0
            ? links.map((link) => (
                <a
                  className="inline-flex items-center gap-1 rounded-full bg-surface-2 px-2 py-0.5 text-ink-2 transition-colors hover:text-ink"
                  href={link.url}
                  key={link.url}
                  rel="noreferrer"
                  target="_blank"
                >
                  {link.icon}
                  {link.label}
                </a>
              ))
            : null
        }
        result={result}
      />
    </ResultArticle>
  );
}
