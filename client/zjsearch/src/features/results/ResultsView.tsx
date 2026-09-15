// SPDX-License-Identifier: Apache-2.0 WITH Commons-Clause-1.0

/**
 * Renders the result collection for the page's detected layout — one switch
 * as the single category → presentation mapping (the mixed-search blocks
 * reuse the same views per category in CategoryBlocks).
 */

import { AppsGrid } from "@/features/results/AppsGrid.tsx";
import { CardList, entriesOf } from "@/features/results/CardList.tsx";
import { CategoryBlocks } from "@/features/results/CategoryBlocks.tsx";
import { DictionaryCard } from "@/features/results/cards/DictionaryCard.tsx";
import { PaperCard } from "@/features/results/cards/PaperCard.tsx";
import { FilesGrid } from "@/features/results/FilesGrid.tsx";
import { ImageGrid } from "@/features/results/image/ImageGrid.tsx";
import type { ResultsLayout } from "@/features/results/layout.ts";
import { MusicGrid } from "@/features/results/MusicGrid.tsx";
import { PackageGrid } from "@/features/results/PackageGrid.tsx";
import { PosterGrid } from "@/features/results/PosterGrid.tsx";
import { ProductGrid } from "@/features/results/ProductGrid.tsx";
import { VideoGrid } from "@/features/results/VideoGrid.tsx";
import type { GlobalData, ResultItem } from "@/lib/types.ts";

export function ResultsView({
  layout,
  results,
  globals,
  selected,
  collapsedBlocks,
  onToggleBlock,
}: {
  layout: ResultsLayout;
  results: ResultItem[];
  globals: GlobalData;
  selected: number;
  collapsedBlocks: Record<string, boolean>;
  onToggleBlock: (key: string) => void;
}) {
  switch (layout.kind) {
    case "images":
      return (
        <div className="mt-4">
          <ImageGrid results={results} />
        </div>
      );
    case "videos":
      return (
        <div className="mt-4">
          <VideoGrid globals={globals} results={results} selected={selected} />
        </div>
      );
    case "music":
      return (
        <div className="mt-4">
          <MusicGrid globals={globals} results={results} selected={selected} />
        </div>
      );
    case "movies":
      return (
        <div className="mt-4">
          <PosterGrid globals={globals} results={results} selected={selected} />
        </div>
      );
    case "apps":
      return (
        <div className="mt-4">
          <AppsGrid globals={globals} results={results} selected={selected} />
        </div>
      );
    case "packages":
      return (
        <div className="mt-4">
          <PackageGrid globals={globals} results={results} selected={selected} />
        </div>
      );
    case "files":
      return (
        <div className="mt-4">
          <FilesGrid globals={globals} results={results} selected={selected} />
        </div>
      );
    case "products":
      return (
        <div className="mt-4">
          <ProductGrid globals={globals} results={results} selected={selected} />
        </div>
      );
    case "dictionary":
      return (
        <CardList
          className="mt-2"
          entries={entriesOf(results)}
          globals={globals}
          renderItem={(result) => <DictionaryCard globals={globals} result={result} />}
          selected={selected}
        />
      );
    case "science":
      return (
        <CardList
          className="mt-2"
          entries={entriesOf(results)}
          globals={globals}
          renderItem={(result) => <PaperCard globals={globals} result={result} />}
          selected={selected}
        />
      );
    case "list":
      // category intent page: a pure relevance-ordered list in which every
      // type keeps its own card - extracting a type into a strip would break
      // the relevance order. space-y keeps highlighted (selected / hovered)
      // cards from touching, matching the mixed-block lists.
      return (
        <div className="mt-2">
          <CardList
            autoOpenMap={layout.autoOpenMap}
            entries={entriesOf(results)}
            globals={globals}
            selected={selected}
          />
        </div>
      );
    case "mixed":
      return (
        <div className="relative mt-2">
          <CategoryBlocks
            autoOpenMap={layout.autoOpenMap}
            collapsedBlocks={collapsedBlocks}
            globals={globals}
            onToggleBlock={onToggleBlock}
            results={results}
            selected={selected}
          />
        </div>
      );
  }
}
