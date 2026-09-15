// SPDX-License-Identifier: Apache-2.0 WITH Commons-Clause-1.0

import { Globe } from "lucide-react";
import type { ReactNode } from "react";
import { useEffect, useState } from "react";
import { EnginesLine, ResultLink } from "@/features/results/cardParts.tsx";
import { TILE_BADGE } from "@/lib/styles.ts";
import type { GlobalData, ResultItem } from "@/lib/types.ts";

/** Corner badge on a media tile — duration, filesize (dark pill, bottom-right). */
export function TileBadge({ children, className = "" }: { children: ReactNode; className?: string }) {
  return <span className={`bottom-2 right-2 ${TILE_BADGE} ${className}`}>{children}</span>;
}

/** Source favicon pinned to the bottom-left of a media tile; falls back to the
    placeholder icon when the engine favicon is missing or blocked by the proxy. */
export function TileFavicon({ src }: { src: string }) {
  const [failed, setFailed] = useState(!src);
  if (failed) {
    return (
      <span className="absolute bottom-2 left-2 flex size-6 items-center justify-center rounded-full bg-white ring-1 ring-white/25">
        <Globe className="size-3.5 text-ink-2" />
      </span>
    );
  }
  return (
    <img
      alt=""
      className="absolute bottom-2 left-2 size-6 rounded-full bg-white object-contain ring-1 ring-white/25"
      decoding="async"
      loading="lazy"
      onError={() => setFailed(true)}
      src={src}
    />
  );
}

/** Compact engine attribution for tile views: the unified EnginesLine
    ([score] [first engine] [+N]); the cached link is icon-only so the row
    stays on one line. */
export function TileEngines({ result }: { result: ResultItem }) {
  return <EnginesLine compact result={result} />;
}

/** Tile thumbnail with graceful failure: a missing/broken/hung thumbnail
    (12s grace) renders the caller's placeholder glyph instead of an error
    image. */
export function TileThumb({
  src,
  alt,
  placeholder,
  imgClassName = "size-full object-cover transition-transform duration-300 group-hover:scale-[1.03]",
}: {
  src?: string;
  alt: string;
  placeholder: ReactNode;
  imgClassName?: string;
}) {
  const [failed, setFailed] = useState(!src);
  const [loaded, setLoaded] = useState(false);
  useEffect(() => {
    if (loaded || failed) {
      return;
    }
    const timer = window.setTimeout(() => setFailed(true), 12000);
    return () => window.clearTimeout(timer);
  }, [failed, loaded]);
  if (failed) {
    return <>{placeholder}</>;
  }
  return (
    <img
      alt={alt}
      className={imgClassName}
      decoding="async"
      loading="lazy"
      onError={() => setFailed(true)}
      onLoad={() => setLoaded(true)}
      src={src ?? ""}
    />
  );
}

// ------------------------------------------------- shared grid-cell anatomy

/** Shared grid-cell scaffold — every media/product/package grid renders its
    cells through this wrapper so the hotkey contract (data-hotkey-index +
    selection ring) and the hover surface stay identical everywhere. */
export function TileCell({
  selected,
  hotkeyIndex,
  children,
}: {
  selected?: number;
  hotkeyIndex: number;
  children: ReactNode;
}) {
  return (
    <article
      className={`group -m-2 flex flex-col rounded-2xl p-2 ${
        selected === hotkeyIndex ? "bg-surface ring-1 ring-accent-strong" : ""
      }`}
      data-hotkey-index={hotkeyIndex}
    >
      {children}
    </article>
  );
}

/** Two-line tile title (fixed slots so every cell in a grid aligns). */
export function TileTitle({
  result,
  globals,
  className = "mt-2.5 min-h-[2.75rem]",
}: {
  result: ResultItem;
  globals: GlobalData;
  className?: string;
}) {
  return (
    <h3 className={`line-clamp-2 text-base font-medium leading-snug ${className}`}>
      <ResultLink
        className="text-ink decoration-accent/50 underline-offset-2 hover:text-accent hover:underline"
        globals={globals}
        result={result}
      >
        <span dangerouslySetInnerHTML={{ __html: result.title_html }} dir="auto" />
      </ResultLink>
    </h3>
  );
}

/** Centered disc action on a tile (play / magnet / download): hover-revealed
    emphasis, accent-strong fill on hover with accent-contrast icon. */
export function TileCenterAction({
  label,
  icon,
  href,
  download,
  onClick,
}: {
  label: string;
  icon: ReactNode;
  /** renders an <a> when set (magnet / download); a play <button> otherwise */
  href?: string;
  download?: boolean;
  onClick?: () => void;
}) {
  const className =
    "absolute left-1/2 top-1/2 z-10 grid size-12 -translate-x-1/2 -translate-y-1/2 place-items-center rounded-full bg-black/70 text-white opacity-90 shadow-pop transition-all hover:scale-105 hover:bg-accent-strong hover:text-accent-contrast group-hover:opacity-100";
  if (href !== undefined) {
    return (
      <a
        aria-label={label}
        className={className}
        href={href}
        {...(download ? { download: true } : {})}
        rel={download ? undefined : "noreferrer"}
        title={label}
      >
        {icon}
      </a>
    );
  }
  return (
    <button aria-label={label} className={className} onClick={onClick} title={label} type="button">
      {icon}
    </button>
  );
}
