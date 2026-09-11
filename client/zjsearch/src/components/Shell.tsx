// SPDX-License-Identifier: AGPL-3.0-or-later

import type { MouseEvent, ReactNode } from "react";
import { useT } from "../lib/i18n.ts";
import { useOverlay } from "../lib/overlay.tsx";
import { useRouter } from "../lib/router.tsx";
import type { GlobalData } from "../lib/types.ts";
import { BarChartIcon, HeartIcon, SlidersIcon } from "./icons.tsx";

/** Anchor that performs SPA navigation for internal URLs. */
export function Link({
  href,
  children,
  className,
  ariaLabel,
  title,
  external,
}: {
  href: string;
  children: ReactNode;
  className?: string;
  ariaLabel?: string;
  title?: string;
  external?: boolean;
}) {
  const { navigate } = useRouter();
  const internal = href.startsWith("/") && !external;

  const onClick = (event: MouseEvent<HTMLAnchorElement>) => {
    if (!internal || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey || event.button !== 0) {
      return;
    }
    event.preventDefault();
    navigate(href);
  };

  return (
    <a
      className={className}
      href={href}
      onClick={onClick}
      {...(ariaLabel ? { "aria-label": ariaLabel } : {})}
      {...(title ? { title } : {})}
      {...(external ? { target: "_blank", rel: "noopener noreferrer" } : { rel: "noreferrer" })}
    >
      {children}
    </a>
  );
}

function ProgressBar({ active }: { active: boolean }) {
  if (!active) {
    return null;
  }
  return (
    <div aria-hidden="true" className="fixed inset-x-0 top-0 z-50 h-0.5 overflow-hidden" role="progressbar">
      <div className="h-full w-full origin-left bg-accent-strong animate-progress" />
    </div>
  );
}

const iconBtn =
  "grid size-8 sm:size-9 place-items-center rounded-full text-ink-2 transition-colors hover:bg-surface-2 hover:text-ink";

/** Right-side icon group: Stats / Preferences open as slide-in
    panels (URL unchanged); theme style lives in the preferences panel. */
export function HeaderActions({ globals }: { globals: GlobalData }) {
  const t = useT();
  const { openOverlay } = useOverlay();
  return (
    <div className="flex items-center gap-0.5 sm:gap-1">
      {globals.donation_url ? (
        <a
          aria-label={t("donate")}
          className={iconBtn}
          href={globals.donation_url}
          rel="noreferrer"
          title={t("donate")}
        >
          <HeartIcon className="size-[18px]" />
        </a>
      ) : null}
      {globals.enable_metrics ? (
        <button
          aria-label={t("engine_stats")}
          className={iconBtn}
          onClick={() => {
            openOverlay("/stats", t("engine_stats"));
          }}
          title={t("engine_stats")}
          type="button"
        >
          <BarChartIcon className="size-[18px]" />
        </button>
      ) : null}
      <button
        aria-label={t("preferences")}
        className={iconBtn}
        onClick={() => {
          openOverlay("/preferences", t("preferences"));
        }}
        title={t("preferences")}
        type="button"
      >
        <SlidersIcon className="size-[18px]" />
      </button>
    </div>
  );
}

/** Standalone top bar used by full pages (preferences/stats/info/404). */
function TopNav({ globals, hideBrand = false }: { globals: GlobalData; hideBrand?: boolean }) {
  return (
    <nav className="flex items-center justify-between gap-3 px-4 py-3 sm:px-6">
      {hideBrand ? (
        <span aria-hidden="true" />
      ) : (
        <Link ariaLabel={globals.instance_name} className="shrink-0 select-none" href="/" title={globals.instance_name}>
          <span className="text-xl font-extrabold tracking-tight text-ink">
            {globals.instance_name}
            <span className="text-accent-strong">.</span>
          </span>
        </Link>
      )}
      <HeaderActions globals={globals} />
    </nav>
  );
}

function Footer() {
  const year = new Date().getFullYear();
  return (
    <footer className="mx-auto w-full max-w-5xl px-4 pb-8 text-center text-xs text-ink-3 sm:px-6">
      <p className="leading-5">© {year} Zhijie Online</p>
    </footer>
  );
}

export function Shell({
  globals,
  children,
  variant = "page",
  hideTopNav = false,
  embedded = false,
}: {
  globals: GlobalData;
  children: ReactNode;
  variant?: "page" | "hero";
  /** results page renders the actions inside its own header */
  hideTopNav?: boolean;
  /** panel mode: page content only, no top bar / footer */
  embedded?: boolean;
}) {
  const { loading } = useRouter();
  if (embedded) {
    return (
      <div className="relative">
        <ProgressBar active={loading} />
        {children}
      </div>
    );
  }
  return (
    <div className={`flex min-h-dvh flex-col ${variant === "hero" ? "" : ""}`}>
      <ProgressBar active={loading} />
      {hideTopNav ? null : <TopNav globals={globals} hideBrand={variant === "hero"} />}
      <div className={`flex flex-1 flex-col ${variant === "hero" ? "justify-center" : ""}`}>{children}</div>
      <Footer />
    </div>
  );
}
