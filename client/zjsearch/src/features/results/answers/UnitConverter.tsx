// SPDX-License-Identifier: Apache-2.0 WITH Commons-Clause-1.0

/** Interactive unit converter (the "unit_converter" plugin): Kagi-style twin
    panels — from/to unit dropdowns with live recomputation and a swap
    button.  The server payload lists the sibling units of the resolved
    dimension (same SI unit); special (callable) converters — °C / °F / Bft —
    are implemented here, the server can't ship them as factors. */

import { ArrowLeftRight } from "lucide-react";
import { useMemo, useState } from "react";
import { Dropdown, type DropdownOption } from "@/components/Dropdown.tsx";
import { useT } from "@/lib/i18n.ts";
import type { LegacyAnswerData, UnitEntry } from "@/lib/types.ts";

type UnitConversionData = Extract<LegacyAnswerData, { kind: "unit_conversion" }>;

/** Beaufort scale in m/s — mirrors searx/wikidata_units.Beaufort. */
const BFT_SCALE = [0.2, 1.5, 3.3, 5.4, 7.9, 10.7, 13.8, 17.1, 20.7, 24.4, 28.4, 32.6, 32.7, 41.1, 45.8, 50.8, 55.6];

const SPECIAL_UNITS: Record<string, { toSI: (v: number) => number; fromSI: (v: number) => number }> = {
  "°C": { toSI: (v) => v + 273.15, fromSI: (v) => v - 273.15 },
  "°F": { toSI: (v) => ((v + 459.67) * 5) / 9, fromSI: (v) => (v * 9) / 5 - 459.67 },
  Bft: {
    toSI: (v) => {
      const idx = Math.round(v);
      return idx >= 0 && idx <= 16 ? (BFT_SCALE[idx] ?? Number.NaN) : Number.NaN;
    },
    fromSI: (v) => {
      const idx = BFT_SCALE.findIndex((mps) => mps >= v);
      return idx === -1 ? 16 : idx;
    },
  },
};

function toSI(value: number, unit: UnitEntry): number {
  if (unit.special) {
    return SPECIAL_UNITS[unit.symbol]?.toSI(value) ?? Number.NaN;
  }
  return value * (unit.to_si ?? 1);
}

function fromSI(value: number, unit: UnitEntry): number {
  if (unit.special) {
    return SPECIAL_UNITS[unit.symbol]?.fromSI(value) ?? Number.NaN;
  }
  return value / (unit.to_si ?? 1);
}

/** Same number language as the server: grouping, up to 10 decimals. */
function formatValue(value: number, maxFraction = 10): string {
  if (!Number.isFinite(value)) {
    return "–";
  }
  return new Intl.NumberFormat("en-US", { maximumFractionDigits: maxFraction }).format(value);
}

function parseValue(raw: string): number | null {
  const cleaned = raw.replaceAll(",", "").trim();
  if (!cleaned) {
    return null;
  }
  const value = Number(cleaned);
  return Number.isFinite(value) ? value : null;
}

export function UnitConverterAnswer({ data }: { data: UnitConversionData }) {
  const t = useT();
  const units = useMemo<UnitEntry[]>(
    () => [...(data.units ?? [])].sort((a, b) => a.symbol.localeCompare(b.symbol)),
    [data.units],
  );
  const [fromUnit, setFromUnit] = useState(data.from_unit);
  const [toUnit, setToUnit] = useState(data.to_unit);
  const [fromInput, setFromInput] = useState(data.from_value);

  const fromEntry = units.find((unit) => unit.symbol === fromUnit);
  const toEntry = units.find((unit) => unit.symbol === toUnit);
  const value = parseValue(fromInput);
  const converted = value !== null && fromEntry && toEntry ? fromSI(toSI(value, fromEntry), toEntry) : null;

  // "1 X = Y Z" only makes sense for linear (factor) dimensions, not for
  // affine ones like temperature
  const linear = Boolean(fromEntry && toEntry && !fromEntry.special && !toEntry.special);
  const rate = linear && fromEntry && toEntry ? formatValue(fromSI(toSI(1, fromEntry), toEntry), 6) : null;
  const rateInverse = linear && fromEntry && toEntry ? formatValue(fromSI(toSI(1, toEntry), fromEntry), 6) : null;

  const swap = () => {
    setFromUnit(toUnit);
    setToUnit(fromUnit);
    // carry the converted value over, like Kagi's swap
    if (converted !== null) {
      setFromInput(formatValue(converted));
    }
  };

  const unitOptions: DropdownOption[] = units.map((unit) => ({ value: unit.symbol, label: unit.symbol }));

  return (
    <div className="animate-fade-up rounded-2xl border border-accent/25 bg-accent-soft/50 px-4 py-4">
      <div className="grid items-stretch gap-2 md:grid-cols-[1fr_auto_1fr] md:gap-3">
        <div className="rounded-2xl border border-line bg-surface px-4 py-3">
          <Dropdown
            align="start"
            ariaLabel={t("unit_from")}
            onChange={setFromUnit}
            options={unitOptions}
            value={fromUnit}
            variant="boxed"
          />
          <input
            aria-label={t("unit_value")}
            className="mt-2 w-full min-w-0 bg-transparent text-end text-4xl font-semibold tabular-nums text-ink outline-none"
            dir="ltr"
            inputMode="decimal"
            onChange={(event) => {
              setFromInput(event.target.value);
            }}
            spellCheck={false}
            value={fromInput}
          />
          <p className="mt-2 truncate border-t border-line pt-2 text-xs text-ink-3" dir="ltr">
            {rate !== null ? `1 ${fromUnit} = ${rate} ${toUnit}` : "\u00A0"}
          </p>
        </div>
        <div className="grid place-items-center">
          <button
            aria-label={t("unit_swap")}
            className="grid size-9 place-items-center rounded-full text-ink-2 transition-colors hover:bg-surface-2 hover:text-ink"
            onClick={swap}
            title={t("unit_swap")}
            type="button"
          >
            <ArrowLeftRight aria-hidden="true" className="size-4.5 max-md:rotate-90" />
          </button>
        </div>
        <div className="rounded-2xl border border-line bg-surface px-4 py-3">
          <Dropdown
            align="start"
            ariaLabel={t("unit_to")}
            onChange={setToUnit}
            options={unitOptions}
            value={toUnit}
            variant="boxed"
          />
          <p className="mt-2 min-h-10 truncate text-end text-4xl font-semibold tabular-nums text-ink" dir="ltr">
            {converted === null ? "–" : formatValue(converted)}
          </p>
          <p className="mt-2 truncate border-t border-line pt-2 text-xs text-ink-3" dir="ltr">
            {rateInverse !== null ? `1 ${toUnit} = ${rateInverse} ${fromUnit}` : "\u00A0"}
          </p>
        </div>
      </div>
    </div>
  );
}
