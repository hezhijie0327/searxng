// SPDX-License-Identifier: Apache-2.0 WITH Commons-Clause-1.0

/**
 * Kagi-style image lightbox: image centered, prev/next round buttons, bottom
 * bar with actions and metadata, keyboard navigation, touch swipe, zoom &
 * drag-pan, and an #image-viewer hash so the back button dismisses it.
 */

import { Award, ChevronLeft, ChevronRight, Download, ExternalLink, ImageOff, Server, X } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";
import { formatScore } from "@/lib/format.ts";
import { useT } from "@/lib/i18n.ts";
import { newTabLinkProps } from "@/lib/link.ts";
import { useSettings } from "@/lib/settings.ts";
import type { ResultItem } from "@/lib/types.ts";

const IMAGE_VIEWER_HASH = "#image-viewer";

function ProgressiveImage({ thumbnail, full, alt }: { thumbnail: string; full: string; alt: string }) {
  const [src, setSrc] = useState(thumbnail);
  useEffect(() => {
    setSrc(thumbnail);
    if (!full || full === thumbnail) {
      return;
    }
    const image = new Image();
    const timer = window.setTimeout(() => {
      image.onload = () => {
        setSrc(full);
      };
      image.onerror = () => {
        setSrc(thumbnail);
      };
      image.src = full;
    }, 600);
    return () => {
      window.clearTimeout(timer);
      image.onload = null;
      image.onerror = null;
    };
  }, [thumbnail, full]);
  const [failed, setFailed] = useState(!src);
  if (failed) {
    return (
      <div className="flex h-[60vh] w-full items-center justify-center text-zinc-500">
        <ImageOff className="size-10" />
      </div>
    );
  }
  return (
    <img
      alt={alt}
      className="max-h-[76vh] max-w-full rounded-md object-contain select-none"
      draggable={false}
      onError={() => {
        setFailed(true);
      }}
      src={src}
    />
  );
}

export function Lightbox({
  results,
  index,
  onClose,
  onNavigate,
}: {
  results: ResultItem[];
  index: number;
  onClose: () => void;
  onNavigate: (index: number) => void;
}) {
  const t = useT();
  const settings = useSettings();
  const result = results[index];
  const touchStartX = useRef<number | null>(null);
  const stageRef = useRef<HTMLDivElement | null>(null);
  const [zoom, setZoom] = useState(1);
  const [offset, setOffset] = useState({ x: 0, y: 0 });
  const [dragging, setDragging] = useState(false);
  const dragStart = useRef<{ x: number; y: number; ox: number; oy: number } | null>(null);
  const [enginesExpanded, setEnginesExpanded] = useState(false);

  const resetZoom = useCallback(() => {
    setZoom(1);
    setOffset({ x: 0, y: 0 });
  }, []);

  // wheel zoom, non-passive so the page never scrolls behind the viewer
  useEffect(() => {
    const el = stageRef.current;
    if (!el) {
      return;
    }
    const onWheel = (event: WheelEvent) => {
      event.preventDefault();
      setZoom((prev) => {
        // allow zooming out to 50% so small thumbnails can shrink to context
        const next = Math.min(5, Math.max(0.5, prev * (event.deltaY < 0 ? 1.15 : 1 / 1.15)));
        return Math.round(next * 100) / 100;
      });
    };
    el.addEventListener("wheel", onWheel, { passive: false });
    return () => {
      el.removeEventListener("wheel", onWheel);
    };
  }, []);

  // reset whenever another image is opened (resetZoom itself is stable)
  // biome-ignore lint/correctness/useExhaustiveDependencies: index change must re-run the reset
  useEffect(() => {
    resetZoom();
  }, [index, resetZoom]);

  const close = useCallback(
    (viaUser: boolean) => {
      if (viaUser && window.location.hash === IMAGE_VIEWER_HASH) {
        history.back(); // hashchange listener calls onClose
      } else {
        onClose();
      }
    },
    [onClose],
  );

  const nav = useCallback(
    (delta: number) => {
      resetZoom();
      const next = (index + delta + results.length) % results.length;
      onNavigate(next);
    },
    [index, results.length, onNavigate, resetZoom],
  );

  useEffect(() => {
    if (window.location.hash !== IMAGE_VIEWER_HASH) {
      window.location.hash = "image-viewer";
    }
    const onHashChange = () => {
      if (window.location.hash !== IMAGE_VIEWER_HASH) {
        onClose();
      }
    };
    window.addEventListener("hashchange", onHashChange);
    return () => {
      window.removeEventListener("hashchange", onHashChange);
    };
  }, [onClose]);

  useEffect(() => {
    const onKeyDown = (event: globalThis.KeyboardEvent) => {
      if (event.key !== "Escape" && event.key !== "ArrowLeft" && event.key !== "ArrowRight") {
        return;
      }
      // capture phase so the results-page hotkeys (arrows paginate) don't
      // also fire while the viewer is open
      event.stopPropagation();
      if (event.key === "Escape") {
        close(true);
      } else if (event.key === "ArrowLeft") {
        nav(-1);
      } else {
        nav(1);
      }
    };
    window.addEventListener("keydown", onKeyDown, { capture: true });
    return () => {
      window.removeEventListener("keydown", onKeyDown, { capture: true });
    };
  }, [close, nav]);

  if (!result) {
    return null;
  }

  const thumbSrc = result.thumbnail_src || result.img_src || "";
  const linkProps = newTabLinkProps(settings.results_on_new_tab);

  let hostname = "";
  if (result.netloc) {
    hostname = result.netloc;
  } else if (result.url) {
    try {
      hostname = new URL(result.url).hostname;
    } catch {
      hostname = "";
    }
  }

  return (
    <div
      aria-modal="true"
      className="fixed inset-0 z-50 flex flex-col bg-[#161616]/97 animate-fade-in"
      onTouchEnd={(event) => {
        const start = touchStartX.current;
        const end = event.changedTouches[0]?.clientX ?? null;
        touchStartX.current = null;
        if (start !== null && end !== null && Math.abs(end - start) > 48) {
          nav(end < start ? 1 : -1);
        }
      }}
      onTouchStart={(event) => {
        touchStartX.current = event.changedTouches[0]?.clientX ?? null;
      }}
      role="dialog"
    >
      {/* top bar */}
      <div className="flex items-center justify-between p-3">
        <span className="text-xs text-zinc-500" dir="ltr">
          {index + 1} / {results.length}
          {zoom !== 1 ? <span className="ms-2 opacity-80">{Math.round(zoom * 100)}%</span> : null}
        </span>
        <button
          aria-label={t("close")}
          className="grid size-9 place-items-center rounded-full text-zinc-300 transition-colors hover:bg-white/10 hover:text-white"
          onClick={() => {
            close(true);
          }}
          type="button"
        >
          <X className="size-5" />
        </button>
      </div>

      {/* image */}
      <div
        className="relative flex min-h-0 flex-1 items-center justify-center overflow-hidden px-4 pb-4"
        ref={stageRef}
      >
        <div
          aria-label={result.title_text}
          className={`flex items-center justify-center ${zoom > 1 ? "touch-none" : ""}`}
          onDoubleClick={() => {
            resetZoom();
          }}
          onPointerDown={(event) => {
            if (zoom <= 1) {
              return;
            }
            dragStart.current = { x: event.clientX, y: event.clientY, ox: offset.x, oy: offset.y };
            setDragging(true);
            event.currentTarget.setPointerCapture(event.pointerId);
          }}
          onPointerMove={(event) => {
            const start = dragStart.current;
            if (!start) {
              return;
            }
            setOffset({ x: start.ox + (event.clientX - start.x), y: start.oy + (event.clientY - start.y) });
          }}
          onPointerUp={(event) => {
            dragStart.current = null;
            setDragging(false);
            event.currentTarget.releasePointerCapture(event.pointerId);
          }}
          role="img"
          style={{
            transform: `translate(${offset.x}px, ${offset.y}px) scale(${zoom})`,
            transition: dragging ? "none" : "transform 150ms ease-out",
            cursor: zoom > 1 ? (dragging ? "grabbing" : "grab") : "zoom-in",
          }}
        >
          <ProgressiveImage alt={result.title_text} full={result.img_src ?? ""} thumbnail={thumbSrc} />
        </div>
        <div className="absolute bottom-2 right-5 flex gap-2">
          <button
            aria-label={t("previous_page")}
            className="grid size-11 place-items-center rounded-full bg-white/10 text-zinc-200 backdrop-blur transition-colors hover:bg-white/20 hover:text-white"
            onClick={() => {
              nav(-1);
            }}
            type="button"
          >
            <ChevronLeft className="size-5" />
          </button>
          <button
            aria-label={t("next_page")}
            className="grid size-11 place-items-center rounded-full bg-white/10 text-zinc-200 backdrop-blur transition-colors hover:bg-white/20 hover:text-white"
            onClick={() => {
              nav(1);
            }}
            type="button"
          >
            <ChevronRight className="size-5" />
          </button>
        </div>
      </div>

      {/* bottom bar: title / actions / metadata - fixed sections so button
          positions never shift with the title length */}
      <div className="flex items-center gap-x-4 border-t border-white/10 bg-[#1c1c1c]/95 px-5 py-3">
        <div className="min-w-0 flex-1">
          {result.url ? (
            <a
              className="block truncate text-sm font-medium text-zinc-100 hover:underline"
              dir="auto"
              href={result.url}
              {...linkProps}
            >
              {result.title_text}
            </a>
          ) : (
            <p className="truncate text-sm font-medium text-zinc-100" dir="auto">
              {result.title_text}
            </p>
          )}
          {hostname ? (
            <p className="truncate text-xs text-zinc-500" dir="ltr">
              {hostname}
            </p>
          ) : null}
          {result.engines.length > 0 ? (
            <div className="mt-2.5 flex flex-wrap items-center gap-1" title={result.engines.join(", ")}>
              {typeof result.score === "number" ? (
                <span
                  className="inline-flex items-center gap-1 rounded-full bg-white/10 px-2 py-0.5 text-[11px] text-zinc-300 tabular-nums"
                  title={t("scores")}
                >
                  <Award className="size-3 shrink-0" />
                  {formatScore(result.score)}
                </span>
              ) : null}
              <span className="inline-flex items-center gap-1 rounded-full bg-white/10 px-2 py-0.5 text-[11px] text-zinc-300">
                <Server className="size-3 shrink-0" />
                {result.engines[0]}
              </span>
              {enginesExpanded
                ? result.engines.slice(1).map((engine) => (
                    <span
                      className="inline-flex items-center gap-1 rounded-full bg-white/10 px-2 py-0.5 text-[11px] text-zinc-300"
                      key={engine}
                    >
                      <Server className="size-3 shrink-0" />
                      {engine}
                    </span>
                  ))
                : null}
              {result.engines.length > 1 ? (
                <button
                  aria-expanded={enginesExpanded}
                  className="rounded-full bg-white/10 px-2 py-0.5 text-[11px] text-zinc-300 transition-colors hover:text-white"
                  onClick={() => {
                    setEnginesExpanded((value) => !value);
                  }}
                  type="button"
                >
                  {enginesExpanded ? t("show_less") : `+${result.engines.length - 1}`}
                </button>
              ) : null}
            </div>
          ) : null}
        </div>

        <div className="flex shrink-0 items-center gap-2 text-xs">
          {result.img_src ? (
            <a
              className="inline-flex items-center gap-1.5 rounded-full border border-zinc-600 px-3.5 py-1.5 text-zinc-200 transition-colors hover:border-zinc-400"
              href={result.img_src}
              {...linkProps}
            >
              {t("view_source")}
              <ExternalLink className="size-3.5" />
            </a>
          ) : null}
          {result.img_src ? (
            <a
              className="inline-flex items-center gap-1.5 rounded-full bg-accent-strong px-3.5 py-1.5 font-medium text-accent-contrast transition-colors hover:bg-accent-strong-hover"
              href={result.img_src}
              {...linkProps}
            >
              {t("download")}
              <Download className="size-3.5" />
            </a>
          ) : null}
        </div>

        <div className="hidden shrink-0 items-center gap-x-6 text-xs xl:flex">
          <Label label={t("resolution")} value={result.resolution} />
          <Label label={t("type")} value={result.img_format} />
          <Label label={t("source")} value={result.source} />
        </div>
      </div>
    </div>
  );
}

function Label({ label, value }: { label: string; value: string | null | undefined }) {
  if (!value) {
    return null;
  }
  return (
    <p className="whitespace-nowrap leading-5">
      <span className="text-zinc-500">{label} </span>
      <span className="text-zinc-200">{value}</span>
    </p>
  );
}
