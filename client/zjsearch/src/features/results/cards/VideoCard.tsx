// SPDX-License-Identifier: Apache-2.0 WITH Commons-Clause-1.0

import { Play, X } from "lucide-react";
import { useState } from "react";
import {
  type CardProps,
  EnginesLine,
  MediaPreview,
  MetaLine,
  PrettyUrl,
  ResultArticle,
  ResultLink,
  Thumb,
  Title,
} from "@/features/results/cardParts.tsx";
import { formatLength } from "@/lib/format.ts";
import { useT } from "@/lib/i18n.ts";

export function VideoCard({ eager, result, globals }: CardProps) {
  const t = useT();
  const [previewOpen, setPreviewOpen] = useState(false);
  const hasMedia = Boolean(result.iframe_src);
  return (
    <ResultArticle priority={result.priority}>
      <div className="flex gap-4">
        <div className="min-w-0 flex-1">
          <PrettyUrl globals={globals} result={result} />
          <div className="mt-1">
            <Title globals={globals} result={result} />
          </div>
          <div className="mt-1">
            <MetaLine result={result} />
          </div>
          {!result.thumbnail && hasMedia && previewOpen ? (
            <div className="mt-2 animate-fade-in">
              <MediaPreview src={result.iframe_src ?? ""} video />
            </div>
          ) : null}
          <p
            className="mt-1.5 line-clamp-2 text-sm leading-relaxed text-ink-2"
            dangerouslySetInnerHTML={{ __html: result.content_html || t("no_description") }}
            dir="auto"
          />
        </div>
        {result.thumbnail ? (
          // the player replaces the thumbnail in place - same behaviour as
          // the video grid, never expanding below the text; on phones the
          // wide size would eat the whole text column, so it stays w-40
          <div className={`relative shrink-0 transition-all ${previewOpen && hasMedia ? "sm:w-72" : "w-40"}`}>
            <div className="relative aspect-video overflow-hidden rounded-xl bg-surface-2">
              <ResultLink className="block size-full" globals={globals} result={result}>
                <Thumb
                  alt={result.title_text}
                  className="size-full"
                  eager={eager}
                  lengthDisplay={formatLength(result.length_display, result.length_seconds)}
                  src={result.thumbnail}
                />
              </ResultLink>
              {hasMedia && previewOpen ? (
                <>
                  <iframe
                    allowFullScreen
                    className="absolute inset-0 size-full"
                    referrerPolicy="origin"
                    src={result.iframe_src ?? ""}
                    title={result.title_text}
                  />
                  <button
                    aria-label={t("hide_video")}
                    className="absolute end-1 top-1 z-10 grid size-7 place-items-center rounded-full bg-black/70 text-white transition-colors hover:bg-accent-strong hover:text-accent-contrast"
                    onClick={() => {
                      setPreviewOpen(false);
                    }}
                    title={t("hide_video")}
                    type="button"
                  >
                    <X className="size-3.5" />
                  </button>
                </>
              ) : hasMedia ? (
                <button
                  aria-label={t("play")}
                  className="absolute inset-0 z-10 grid size-full place-items-center rounded-xl bg-black/0 text-white transition-colors hover:bg-black/30"
                  onClick={() => {
                    setPreviewOpen(true);
                  }}
                  title={t("play")}
                  type="button"
                >
                  <span className="grid size-9 place-items-center rounded-full bg-black/70 shadow-pop">
                    <Play className="size-4 translate-x-px" />
                  </span>
                </button>
              ) : null}
            </div>
          </div>
        ) : null}
      </div>
      <EnginesLine result={result} />
    </ResultArticle>
  );
}
