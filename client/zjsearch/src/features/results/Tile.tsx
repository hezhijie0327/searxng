// SPDX-License-Identifier: Apache-2.0 WITH Commons-Clause-1.0

import { Globe } from "lucide-react";
import type { ReactNode } from "react";
import { useEffect, useState } from "react";
import { EnginesLine } from "@/features/results/cardParts.tsx";
import type { ResultItem } from "@/lib/types.ts";

/** Corner badge on a media tile — duration, filesize (dark pill, bottom-right). */
export function TileBadge({ children }: { children: ReactNode }) {
  return (
    <span className="absolute bottom-2 right-2 rounded bg-black/80 px-1.5 py-0.5 text-[11px] font-medium text-white">
      {children}
    </span>
  );
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
    ([score] [first engine] [+N]). */
export function TileEngines({ result }: { result: ResultItem }) {
  return <EnginesLine result={result} />;
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
