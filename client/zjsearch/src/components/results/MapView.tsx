// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * Interactive OSM map for map results.  OpenLayers is lazy-loaded only when
 * the user asks for the map (mirrors the upstream MapView client plugin).
 */

import type Feature from "ol/Feature.js";
import type BaseLayer from "ol/layer/Base.js";
import type { default as OlMap } from "ol/Map.js";
import type VectorSource from "ol/source/Vector.js";
import { useEffect, useRef, useState } from "react";
import { useT } from "../../lib/i18n.ts";
import { LocationIcon } from "../icons.tsx";

interface MapResultProps {
  longitude?: string;
  latitude?: string;
  boundingbox?: number[];
  geojson?: unknown;
  label: string;
  /** render the map expanded from the start (map-intent searches) */
  autoOpen?: boolean;
}

export function MapResult({ longitude, latitude, boundingbox, geojson, label, autoOpen = false }: MapResultProps) {
  const [open, setOpen] = useState(autoOpen);
  const t = useT();
  const containerRef = useRef<HTMLDivElement>(null);
  const hasMap = Boolean(longitude && latitude) || Boolean(boundingbox?.length) || Boolean(geojson);

  useEffect(() => {
    if (!open || !containerRef.current) {
      return;
    }
    let disposed = false;
    let instance: OlMap | null = null;

    void (async () => {
      const container = containerRef.current;
      if (!container) {
        return;
      }
      const { default: OlMap } = await import("ol/Map.js");
      const { default: View } = await import("ol/View.js");
      const { default: TileLayer } = await import("ol/layer/Tile.js");
      const { default: OSM } = await import("ol/source/OSM.js");
      const { fromLonLat } = await import("ol/proj.js");
      if (disposed) {
        return;
      }

      const layers: BaseLayer[] = [new TileLayer({ source: new OSM() })];
      let center = fromLonLat([0, 0]);
      let vectorSource: VectorSource<Feature> | null = null;

      if (geojson) {
        const { default: VectorLayer } = await import("ol/layer/Vector.js");
        const { default: VectorSource } = await import("ol/source/Vector.js");
        const { default: GeoJSON } = await import("ol/format/GeoJSON.js");
        const { default: StrokeStyle } = await import("ol/style/Stroke.js");
        const { default: FillStyle } = await import("ol/style/Fill.js");
        const { default: Style } = await import("ol/style/Style.js");
        const format = new GeoJSON();
        vectorSource = new VectorSource({
          features: format.readFeatures(geojson, {
            dataProjection: "EPSG:4326",
            featureProjection: "EPSG:3857",
          }) as Feature[],
        });
        layers.push(
          new VectorLayer({
            source: vectorSource,
            style: new Style({
              stroke: new StrokeStyle({ color: "#5457d6", width: 2 }),
              fill: new FillStyle({ color: "rgba(84, 87, 214, 0.15)" }),
            }),
          }),
        );
      }

      if (longitude && latitude) {
        const lon = Number(longitude);
        const lat = Number(latitude);
        if (Number.isFinite(lon) && Number.isFinite(lat)) {
          center = fromLonLat([lon, lat]);
        }
      }

      const map = new OlMap({
        target: container,
        layers,
        view: new View({ center, zoom: 12, maxZoom: 16 }),
      });
      instance = map;

      const fitExtent = (extent: number[]) => {
        if (extent.length === 4 && extent.every((value) => Number.isFinite(value))) {
          map.getView().fit(extent, { size: map.getSize(), padding: [24, 24, 24, 24], maxZoom: 16 });
        }
      };

      if (boundingbox && boundingbox.length === 4) {
        const [minLon, minLat, maxLon, maxLat] = boundingbox as [number, number, number, number];
        const min = fromLonLat([minLon, minLat]);
        const max = fromLonLat([maxLon, maxLat]);
        fitExtent([Number(min[0]), Number(min[1]), Number(max[0]), Number(max[1])]);
      } else if (vectorSource) {
        const applyFit = () => {
          fitExtent(vectorSource?.getExtent() ?? []);
        };
        vectorSource.on("change", applyFit);
      }
    })();

    return () => {
      disposed = true;
      instance?.setTarget(undefined);
    };
  }, [open, longitude, latitude, boundingbox, geojson]);

  if (!hasMap) {
    return null;
  }

  return (
    <div className="mt-2">
      <button
        className="inline-flex items-center gap-1.5 rounded-full bg-surface-2 px-3 py-1 text-xs text-ink-2 transition-colors hover:text-ink"
        onClick={() => {
          setOpen((prev) => !prev);
        }}
        type="button"
      >
        <LocationIcon className="size-3.5" />
        {open ? t("hide_map") : label}
      </button>
      {open ? (
        <div
          className="mt-2 h-72 w-full overflow-hidden rounded-xl border border-line animate-fade-in"
          ref={containerRef}
        />
      ) : null}
    </div>
  );
}
