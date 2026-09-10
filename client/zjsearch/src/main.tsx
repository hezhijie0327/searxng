// SPDX-License-Identifier: AGPL-3.0-or-later

import { createRoot } from "react-dom/client";
import { App } from "./app.tsx";
import { extractPageData, parseClientSettings, parseEmbeddedPageData } from "./lib/pageData.ts";
import type { AnyPageData } from "./lib/types.ts";
import "./styles/global.css";

/**
 * Prod: parse the payload embedded by the server shell.
 * Dev (vite dev server): fetch it through the dev proxy.
 */
async function bootData(): Promise<AnyPageData | null> {
  const embedded = parseEmbeddedPageData();
  if (embedded) {
    return embedded;
  }
  if (import.meta.env.DEV) {
    try {
      const resp = await fetch(window.location.href, { headers: { Accept: "text/html" } });
      return extractPageData(await resp.text());
    } catch {
      return null;
    }
  }
  return null;
}

async function bootstrap(): Promise<void> {
  const container = document.getElementById("app");
  if (!container) {
    return;
  }
  const initialData = await bootData();
  const settings = parseClientSettings();
  createRoot(container).render(<App initialData={initialData} settings={settings} />);
}

void bootstrap();
