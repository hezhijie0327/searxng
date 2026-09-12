// SPDX-License-Identifier: Apache-2.0 WITH Commons-Clause-1.0

import type { ReactNode } from "react";
import { THEME_STATIC } from "../../lib/constants.ts";

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
  return (
    <img
      alt=""
      className="absolute bottom-2 left-2 size-6 rounded-full bg-white ring-1 ring-white/25"
      decoding="async"
      loading="lazy"
      onError={(event) => {
        event.currentTarget.src = `${THEME_STATIC}/img/empty_favicon.svg`;
      }}
      src={src}
    />
  );
}
