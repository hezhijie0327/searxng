// SPDX-License-Identifier: Apache-2.0 WITH Commons-Clause-1.0

import { ExternalLink, Server } from "lucide-react";
import { useEffect, useState } from "react";
import { Link, Shell } from "@/components/Shell.tsx";
import { useOverlay } from "@/features/overlay/OverlayProvider.tsx";
import { useT } from "@/lib/i18n.ts";
import type { GlobalData, InfoPageData } from "@/lib/types.ts";

/** Minimal config shape the about card reads from GET /config. */
interface InstanceConfig {
  version: string;
  categories: string[];
  engines: Array<{ enabled: boolean }>;
  plugins: Array<{ enabled: boolean }>;
}

/** One-line instance summary for the about page (engines / categories /
    plugins counts), fetched from /config when the card mounts. Stays hidden
    when the request fails — it is decoration, not content. */
function AboutInstance({ globals }: { globals: GlobalData }) {
  const t = useT();
  const [config, setConfig] = useState<InstanceConfig | null>(null);
  useEffect(() => {
    const controller = new AbortController();
    void fetch("/config", { headers: { Accept: "application/json" }, signal: controller.signal })
      .then((resp) => {
        if (!resp.ok) {
          throw new Error(`HTTP ${resp.status}`);
        }
        return resp.json() as Promise<InstanceConfig>;
      })
      .then((config) => {
        setConfig(config);
      })
      .catch(() => {
        /* quiet: the card simply stays hidden */
      });
    return () => {
      controller.abort();
    };
  }, []);

  if (!config) {
    return null;
  }
  const engines = config.engines.filter((engine) => engine.enabled).length;
  const plugins = config.plugins.filter((plugin) => plugin.enabled).length;
  return (
    <section className="mt-8 flex flex-wrap items-center gap-x-3 gap-y-1 rounded-2xl border border-line bg-surface px-4 py-3 text-xs text-ink-3 animate-fade-up">
      <span className="inline-flex items-center gap-1.5 font-medium text-ink-2">
        <Server className="size-3.5 shrink-0" />
        {globals.instance_name}
      </span>
      <span className="font-mono" dir="ltr">
        v{config.version || globals.version}
      </span>
      <span className="ms-auto inline-flex flex-wrap gap-x-3 gap-y-1">
        <span>
          {engines} {t("engines")}
        </span>
        <span>
          {config.categories.length} {t("categories")}
        </span>
        <span>
          {plugins} {t("plugins")}
        </span>
      </span>
    </section>
  );
}

/** About-page footer: the attribution (powered by), the license document
    link and the compliance links configured on the instance — everything
    that used to sit in the preferences footer, in one place. */
function AboutFooter({ globals }: { globals: GlobalData }) {
  const t = useT();
  const { openDocument } = useOverlay();
  const links: Array<{ label: string; url: string }> = [];
  if (globals.privacypolicy_url) {
    links.push({ label: t("privacypolicy"), url: globals.privacypolicy_url });
  }
  if (globals.contact_url) {
    links.push({ label: t("contact"), url: globals.contact_url });
  }
  if (globals.public_instances_url) {
    links.push({ label: t("public_instances"), url: globals.public_instances_url });
  }
  return (
    <>
      <AboutInstance globals={globals} />
      <footer className="mt-3 space-y-1.5 text-xs text-ink-3">
        <p className="leading-5">
          {t("powered_by")}{" "}
          <a
            className="transition-colors hover:text-accent hover:underline"
            href={globals.git_url}
            rel="noreferrer"
            target="_blank"
          >
            SearXNG
          </a>
        </p>
        <p className="leading-5">
          {t("license")}:{" "}
          <a
            className="cursor-pointer transition-colors hover:text-accent hover:underline"
            href="/static/themes/zjsearch/LICENSE.txt"
            onClick={(event) => {
              event.preventDefault();
              openDocument(t("license"), "/static/themes/zjsearch/LICENSE.txt");
            }}
          >
            Apache-2.0 with Commons Clause v1.0
          </a>
        </p>
        {links.length > 0 ? (
          <div className="flex flex-wrap gap-x-4 gap-y-1 pt-1">
            {links.map((link) => (
              <a
                className="inline-flex items-center gap-1 transition-colors hover:text-ink"
                href={link.url}
                key={link.url}
                rel="noreferrer"
                target="_blank"
              >
                {link.label}
                <ExternalLink className="size-3 shrink-0" />
              </a>
            ))}
          </div>
        ) : null}
      </footer>
    </>
  );
}

export function InfoPage({ data, embedded = false }: { data: InfoPageData; embedded?: boolean }) {
  const globals = data.globals;
  return (
    <Shell embedded={embedded} globals={globals}>
      <main className="mx-auto w-full max-w-3xl flex-1 px-4 pb-16 sm:px-6">
        <nav aria-label="info pages" className="flex flex-wrap gap-1.5 py-5">
          {data.pages.map((page) => {
            const active = page.pagename === data.active_pagename;
            return (
              <Link
                className={`rounded-full border px-3.5 py-1.5 text-[13px] transition-colors ${
                  active
                    ? "border-accent bg-accent-soft font-medium text-accent"
                    : "border-line text-ink-2 hover:border-ink-3 hover:text-ink"
                }`}
                href={`/info/${page.locale}/${page.pagename}`}
                key={`${page.locale}/${page.pagename}`}
              >
                {page.title}
              </Link>
            );
          })}
        </nav>
        <article
          className="prose-basic animate-fade-up"
          dangerouslySetInnerHTML={{ __html: data.active_html }}
          dir="auto"
        />
        {data.active_pagename === "about" ? <AboutFooter globals={globals} /> : null}
      </main>
    </Shell>
  );
}
