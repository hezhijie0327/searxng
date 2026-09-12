// SPDX-License-Identifier: AGPL-3.0-or-later

import { type ReactNode, useEffect, useState } from "react";
import type { DropdownOption } from "../../components/Dropdown.tsx";
import { Dropdown } from "../../components/Dropdown.tsx";
import { AlertIcon, CategoryIcon, ExternalLinkIcon, SparkIcon } from "../../components/icons.tsx";
import { Link } from "../../components/Shell.tsx";
import { loadEngineDescriptions } from "../../lib/engineDescriptions.ts";
import type { EngineEntry } from "../../lib/types.ts";

// ------------------------------------------------------------- row primitives

/** Category selector styled like the results-page category tabs: icon +
    label, selected = accent text with an amber underline. */
export function CategoryTab({
  active,
  category,
  label,
  onClick,
}: {
  active: boolean;
  category: string;
  label: string;
  onClick: () => void;
}) {
  return (
    <button
      aria-pressed={active}
      className={`relative flex items-center gap-1.5 px-4 py-2 text-[13px] transition-colors ${
        active ? "font-medium text-accent" : "text-ink-2 hover:text-ink"
      }`}
      onClick={onClick}
      type="button"
    >
      <CategoryIcon category={category} className="size-3.5 shrink-0" />
      {label}
      <span
        aria-hidden="true"
        className={`absolute inset-x-4 -bottom-0.5 h-0.5 rounded-full bg-accent-strong transition-opacity ${
          active ? "opacity-100" : "opacity-0"
        }`}
      />
    </button>
  );
}

// ------------------------------------------------------------ layout blocks

export function cap(value: string): string {
  return value.charAt(0).toUpperCase() + value.slice(1);
}

export function IconTile({ children }: { children: ReactNode }) {
  return (
    <span className="grid size-10 shrink-0 place-items-center rounded-xl bg-accent-soft text-accent">{children}</span>
  );
}

/** One settings row: icon tile + title/description on the left, control on the right. */
export function SettingRow({
  icon,
  title,
  description,
  children,
  stacked,
}: {
  icon: ReactNode;
  title: string;
  description?: string;
  children: ReactNode;
  stacked?: boolean;
}) {
  if (stacked) {
    return (
      <div className="px-5 py-5 transition-colors hover:bg-surface-2/40 sm:px-6">
        <div className="flex items-center gap-4">
          <IconTile>{icon}</IconTile>
          <div className="min-w-0">
            <p className="text-sm font-medium text-ink">{title}</p>
            {description ? <p className="mt-0.5 text-xs leading-relaxed text-ink-3">{description}</p> : null}
          </div>
        </div>
        <div className="mt-4 sm:pl-14">{children}</div>
      </div>
    );
  }
  return (
    <div className="flex flex-col gap-3 px-5 py-5 transition-colors hover:bg-surface-2/40 sm:flex-row sm:items-center sm:justify-between sm:gap-8 sm:px-6">
      <div className="flex min-w-0 items-center gap-4">
        <IconTile>{icon}</IconTile>
        <div className="min-w-0">
          <p className="text-sm font-medium text-ink">{title}</p>
          {description ? <p className="mt-0.5 text-xs leading-relaxed text-ink-3">{description}</p> : null}
        </div>
      </div>
      <div className="shrink-0">{children}</div>
    </div>
  );
}

export function Card({ children }: { children: ReactNode }) {
  return (
    <div className="divide-y divide-line overflow-hidden rounded-2xl border border-line bg-surface animate-fade-up">
      {children}
    </div>
  );
}

export function Switch({
  checked,
  onChange,
  label,
}: {
  checked: boolean;
  onChange: (checked: boolean) => void;
  label: string;
}) {
  return (
    <button
      aria-checked={checked}
      aria-label={label}
      className={`relative inline-flex h-6 w-11 shrink-0 cursor-pointer items-center rounded-full p-0.5 transition-colors focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent ${
        checked ? "bg-accent-strong" : "bg-surface-2 ring-1 ring-line"
      }`}
      onClick={() => {
        onChange(!checked);
      }}
      role="switch"
      type="button"
    >
      <span
        className={`size-5 rounded-full shadow transition-transform ${
          checked ? "translate-x-5 bg-accent-contrast" : "translate-x-0 bg-ink-3"
        }`}
      />
    </button>
  );
}

export function Select({
  value,
  options,
  onChange,
  ariaLabel,
}: {
  value: string;
  options: DropdownOption[];
  onChange: (value: string) => void;
  ariaLabel?: string;
}) {
  return (
    <div className="w-full sm:w-60">
      <Dropdown align="end" ariaLabel={ariaLabel} onChange={onChange} options={options} value={value} variant="boxed" />
    </div>
  );
}

export function PluginRow({
  plugin,
  enabled,
  onChange,
}: {
  plugin: { id: string; name: string; description: string };
  enabled: boolean;
  onChange: (checked: boolean) => void;
}) {
  return (
    <SettingRow description={plugin.description} icon={<SparkIcon className="size-4.5" />} title={plugin.name}>
      <Switch checked={enabled} label={plugin.name} onChange={onChange} />
    </SettingRow>
  );
}

// ------------------------------------------------------------------ engines

export function reliabilityColor(reliability: number | null): string {
  if (reliability === null) {
    return "text-ink-3";
  }
  if (reliability <= 50) {
    return "text-danger";
  }
  if (reliability < 80) {
    return "text-warning";
  }
  if (reliability < 90) {
    return "text-ink-2";
  }
  return "text-ok";
}

export function EngineTooltip({ engine }: { engine: EngineEntry }) {
  const [desc, setDesc] = useState<{ text: string; source: string } | null>(null);
  useEffect(() => {
    void loadEngineDescriptions().then((map) => {
      const entry = map[engine.name];
      if (entry) {
        setDesc({ text: entry[0], source: entry[1] });
      }
    });
  }, [engine.name]);

  return (
    <div className="pointer-events-none absolute start-0 top-full z-30 mt-1 hidden w-80 rounded-xl border border-line bg-surface p-3 text-xs shadow-pop group-hover/engine:block">
      {desc ? (
        <p className="text-ink-2">
          {desc.text} <i className="text-ink-3">(Source: {desc.source})</i>
        </p>
      ) : (
        <p className="text-ink-3">…</p>
      )}
      {engine.website ? (
        <p className="mt-1.5 truncate">
          <a
            className="inline-flex items-center gap-1 text-accent hover:underline"
            href={engine.website}
            rel="noreferrer"
            target="_blank"
          >
            {engine.website}
            <ExternalLinkIcon className="size-3" />
          </a>
        </p>
      ) : null}
      {engine.enable_http ? (
        <p className="mt-1.5 inline-flex items-center gap-1 text-warning">
          <AlertIcon className="size-3.5" /> No HTTPS
        </p>
      ) : null}
      <p className="mt-1.5 flex flex-wrap gap-1">
        <span className="text-ink-3">!bang:</span>
        {[engine.name, engine.shortcut].map((bang) => (
          <code className="rounded bg-surface-2 px-1" key={bang}>
            !{bang.replaceAll(" ", "_")}
          </code>
        ))}
      </p>
      {engine.errors.length > 0 ? (
        <p className="mt-1.5">
          <Link className="text-accent hover:underline" href={`/stats?engine=${encodeURIComponent(engine.name)}`}>
            View error logs and submit a bug report
          </Link>
        </p>
      ) : null}
    </div>
  );
}
