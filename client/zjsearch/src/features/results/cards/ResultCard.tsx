// SPDX-License-Identifier: Apache-2.0 WITH Commons-Clause-1.0

import { memo } from "react";
import type { CardProps } from "@/features/results/cardParts.tsx";
import { CodeCard } from "@/features/results/cards/CodeCard.tsx";
import { DefaultCard } from "@/features/results/cards/DefaultCard.tsx";
import { FileCard } from "@/features/results/cards/FileCard.tsx";
import { KeyValueCard } from "@/features/results/cards/KeyValueCard.tsx";
import { MapCard } from "@/features/results/cards/MapCard.tsx";
import { PackageCard } from "@/features/results/cards/PackageCard.tsx";
import { PaperCard } from "@/features/results/cards/PaperCard.tsx";
import { ProductCard } from "@/features/results/cards/ProductCard.tsx";
import { TorrentCard } from "@/features/results/cards/TorrentCard.tsx";
import { VideoCard } from "@/features/results/cards/VideoCard.tsx";

/** Kagi-style video tiles for video-only result pages.  Tiles with an
    embeddable source get a Spotify-style hover play button that expands the
    player in place.  Cells carry data-hotkey-index so the results hotkeys
    can walk the grid like the list layouts. */
export const ResultCard = memo(function ResultCard(props: CardProps) {
  const { result } = props;
  switch (result.template) {
    case "images":
      return <DefaultCard {...props} />;
    case "videos":
      return <VideoCard {...props} />;
    case "torrent":
      return <TorrentCard {...props} />;
    case "map":
      return <MapCard {...props} />;
    case "paper":
      return <PaperCard {...props} />;
    case "packages":
      return <PackageCard {...props} />;
    case "code":
      return <CodeCard {...props} />;
    case "file":
      return <FileCard {...props} />;
    case "keyvalue":
      return <KeyValueCard {...props} />;
    case "products":
      return <ProductCard {...props} />;
    default:
      return <DefaultCard {...props} />;
  }
});
// memo: list items only change selection styling on their wrapper (the page
// renders the ring there), so unchanged props let hotkey navigation skip the
// whole card subtree
