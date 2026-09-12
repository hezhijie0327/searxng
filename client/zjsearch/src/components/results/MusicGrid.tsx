// SPDX-License-Identifier: AGPL-3.0-or-later

/** Music-intent layout: square album-art tiles mirroring the video grid.
    Playable results swap the tile for an in-place player - raw audio
    streams get a custom mini player (blur + play/pause + seek), embeddable
    sources play inside the tile like videos do. */

import { useEffect, useRef, useState } from "react";
import { formatDate, formatLength } from "../../lib/format.ts";
import { useT } from "../../lib/i18n.ts";
import type { GlobalData, ResultItem } from "../../lib/types.ts";
import { CalendarIcon, CloseIcon, MusicIcon, PauseIcon, PlayIcon } from "../icons.tsx";
import { ResultLink, THEME_STATIC } from "./cards.tsx";

function formatClock(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds < 0) {
    return "--:--";
  }
  const total = Math.round(seconds);
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  const secs = String(total % 60).padStart(2, "0");
  return hours > 0 ? `${hours}:${String(minutes).padStart(2, "0")}:${secs}` : `${minutes}:${secs}`;
}

/** In-tile mini player for raw audio streams.  Reports playback failure so
    the grid can fall back to the embeddable player when one exists. */
function AudioTilePlayer({ src, onClose, onError }: { src: string; onClose: () => void; onError: () => void }) {
  const t = useT();
  const audioRef = useRef<HTMLAudioElement>(null);
  const [playing, setPlaying] = useState(false);
  const [time, setTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const hasDuration = Number.isFinite(duration) && duration > 0;

  useEffect(() => {
    // the player mounts right after a click, so user activation allows this
    audioRef.current?.play().catch(() => {});
  }, []);

  const toggle = () => {
    const audio = audioRef.current;
    if (!audio) {
      return;
    }
    if (audio.paused) {
      audio.play().catch(() => {});
    } else {
      audio.pause();
    }
  };

  return (
    <div className="absolute inset-0 z-10 flex animate-fade-in flex-col overflow-hidden rounded-xl border border-line bg-black/75 text-white backdrop-blur-md">
      <audio
        onError={onError}
        onLoadedMetadata={(event) => {
          setDuration(event.currentTarget.duration);
        }}
        onPause={() => {
          setPlaying(false);
        }}
        onPlay={() => {
          setPlaying(true);
        }}
        onTimeUpdate={(event) => {
          setTime(event.currentTarget.currentTime);
        }}
        preload="metadata"
        ref={audioRef}
        src={src}
      />
      <button
        aria-label={t("close")}
        className="absolute end-2 top-2 z-10 grid size-7 place-items-center rounded-full bg-black/70 text-white transition-colors hover:bg-accent-strong hover:text-ink"
        onClick={onClose}
        title={t("close")}
        type="button"
      >
        <CloseIcon className="size-3.5" />
      </button>
      <div className="flex flex-1 items-center justify-center">
        <button
          aria-label={playing ? t("pause") : t("play")}
          className="grid size-14 place-items-center rounded-full bg-white text-black shadow-pop transition-transform hover:scale-105"
          onClick={toggle}
          title={playing ? t("pause") : t("play")}
          type="button"
        >
          {playing ? <PauseIcon className="size-6" /> : <PlayIcon className="size-6 translate-x-0.5" />}
        </button>
      </div>
      <div className="flex items-center gap-2 px-3 pb-3 text-[11px] font-medium tabular-nums">
        <span>{formatClock(time)}</span>
        <input
          aria-label={t("length")}
          className="w-full accent-white disabled:opacity-40"
          disabled={!hasDuration}
          max={hasDuration ? duration : 1}
          min={0}
          onChange={(event) => {
            const audio = audioRef.current;
            const value = Number(event.target.value);
            if (audio && Number.isFinite(value)) {
              audio.currentTime = value;
              setTime(value);
            }
          }}
          step={0.1}
          type="range"
          value={hasDuration ? Math.min(time, duration) : 0}
        />
        <span>{formatClock(duration)}</span>
      </div>
    </div>
  );
}

function EmbedTile({ src, title, onClose }: { src: string; title: string; onClose: () => void }) {
  const t = useT();
  return (
    <div className="absolute inset-0 z-10 animate-fade-in overflow-hidden rounded-xl border border-line bg-black">
      <iframe allowFullScreen className="size-full" referrerPolicy="origin" src={src} title={title} />
      <button
        aria-label={t("close")}
        className="absolute end-2 top-2 z-20 grid size-7 place-items-center rounded-full bg-black/70 text-white transition-colors hover:bg-accent-strong hover:text-ink"
        onClick={onClose}
        title={t("close")}
        type="button"
      >
        <CloseIcon className="size-3.5" />
      </button>
    </div>
  );
}

export function MusicGrid({
  results,
  globals,
  selected,
  indexOffset = 0,
}: {
  results: ResultItem[];
  globals: GlobalData;
  selected?: number;
  /** hotkey indices are page-global: offset by the grid's first result index */
  indexOffset?: number;
}) {
  const t = useT();
  const [playing, setPlaying] = useState<number | null>(null);
  const [mode, setMode] = useState<"audio" | "embed">("audio");
  const cells = results.map((result, index) => {
    const length = formatLength(result.length_display, result.length_seconds);
    const isPlaying = playing === index;
    const hotkeyIndex = indexOffset + index;
    const audioSrc = result.audio_src || "";
    const embedSrc = result.iframe_src || "";
    const playable = Boolean(audioSrc || embedSrc);
    return (
      <article
        className={`group -m-2 rounded-2xl p-2 ${selected === hotkeyIndex ? "bg-surface ring-1 ring-accent-strong" : ""}`}
        data-hotkey-index={hotkeyIndex}
        key={`${result.url}-${index}`}
      >
        <div className="relative">
          <ResultLink
            className="relative block aspect-square overflow-hidden rounded-xl bg-surface-2"
            globals={globals}
            result={result}
          >
            {result.thumbnail ? (
              <img
                alt={result.title_text}
                className="size-full object-cover transition-transform duration-300 group-hover:scale-[1.03]"
                decoding="async"
                loading="lazy"
                onError={(event) => {
                  event.currentTarget.src = `${THEME_STATIC}/img/img_load_error.svg`;
                }}
                src={result.thumbnail}
              />
            ) : (
              <span className="grid size-full place-items-center bg-gradient-to-br from-surface-2 to-surface text-ink-3">
                <MusicIcon className="size-10" />
              </span>
            )}
            {length ? (
              <span className="absolute bottom-2 right-2 rounded bg-black/80 px-1.5 py-0.5 text-[11px] font-medium text-white">
                {length}
              </span>
            ) : null}
            {result.favicon ? (
              <img
                alt=""
                className="absolute bottom-2 left-2 size-6 rounded-full bg-white ring-1 ring-white/25"
                decoding="async"
                loading="lazy"
                onError={(event) => {
                  event.currentTarget.src = `${THEME_STATIC}/img/empty_favicon.svg`;
                }}
                src={result.favicon}
              />
            ) : null}
          </ResultLink>
          {playable && isPlaying ? (
            mode === "audio" && audioSrc ? (
              <AudioTilePlayer
                onClose={() => {
                  setPlaying(null);
                }}
                onError={() => {
                  // raw stream failed - fall back to the embed when one exists
                  if (embedSrc) {
                    setMode("embed");
                  } else {
                    setPlaying(null);
                  }
                }}
                src={audioSrc}
              />
            ) : (
              <EmbedTile
                onClose={() => {
                  setPlaying(null);
                }}
                src={embedSrc}
                title={result.title_text}
              />
            )
          ) : null}
          {playable && !isPlaying ? (
            <button
              aria-label={t("play")}
              className="absolute left-1/2 top-1/2 z-10 grid size-12 -translate-x-1/2 -translate-y-1/2 place-items-center rounded-full bg-black/60 text-white opacity-85 shadow-pop transition-all hover:scale-105 hover:bg-accent-strong hover:text-ink group-hover:opacity-100"
              onClick={() => {
                setMode(audioSrc ? "audio" : "embed");
                setPlaying(index);
              }}
              title={t("play")}
              type="button"
            >
              <PlayIcon className="size-5 translate-x-px" />
            </button>
          ) : null}
        </div>
        <h3 className="mt-2.5 line-clamp-2 min-h-[2.75rem] text-base font-medium leading-snug">
          <ResultLink
            className="text-ink decoration-accent/50 underline-offset-2 hover:text-accent hover:underline"
            globals={globals}
            result={result}
          >
            <span dangerouslySetInnerHTML={{ __html: result.title_html }} dir="auto" />
          </ResultLink>
        </h3>
        <div className="mt-1.5 flex items-center justify-between gap-3 text-xs text-ink-3">
          <span className="truncate" dir="auto">
            {result.author || result.engines[0]}
          </span>
          <span className="flex shrink-0 items-center gap-1">
            {result.published_date ? (
              <>
                <CalendarIcon className="size-3" />
                {formatDate(result.published_date)}
              </>
            ) : null}
          </span>
        </div>
      </article>
    );
  });
  return <div className="grid grid-cols-2 gap-x-4 gap-y-8 sm:grid-cols-3 xl:grid-cols-4">{cells}</div>;
}
