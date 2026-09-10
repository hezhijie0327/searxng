// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * zjsearch theme -- Vite build configuration.
 *
 * Sources live in client/zjsearch, build output goes straight into the
 * served static folder searx/static/themes/zjsearch (same layout as the
 * upstream "simple" theme workspace).
 */

import { resolve } from "node:path";
import tailwindcss from "@tailwindcss/vite";
import browserslistToEsbuild from "browserslist-to-esbuild";
import manifest from "./package.json" with { type: "json" };
import { plgAssets } from "./tools/assets.ts";

const ROOT = "../../"; // root of the git repository

const PATH = {
  brand: "src/brand/",
  dist: resolve(ROOT, "searx/static/themes/zjsearch/"),
  src: "src/",
} as const;

// local SearXNG instance used by `npm run dev` (start it with: make run)
const DEV_BACKEND = process.env.ZJSEARCH_BACKEND || "http://127.0.0.1:8888";

export default {
  base: "./",

  publicDir: "static/",

  server: {
    port: 5175,
    proxy: Object.fromEntries(
      [
        "/search",
        "/",
        "/autocompleter",
        "/preferences",
        "/clear_cookies",
        "/image_proxy",
        "/favicon_proxy",
        "/config",
        "/stats",
        "/info",
        "/about",
        "/engine_descriptions.json",
        "/opensearch.xml",
        "/manifest.json",
        "/client",
        "/favicon.ico",
        "/logo",
      ].map((path) => [path, { target: DEV_BACKEND, changeOrigin: true }]),
    ),
  },

  build: {
    target: browserslistToEsbuild(manifest.browserslist),
    assetsDir: "",
    outDir: PATH.dist,
    manifest: "manifest.json",
    emptyOutDir: true,
    sourcemap: true,
    rollupOptions: {
      input: {
        app: `${PATH.src}/main.tsx`,
      },
      output: {
        entryFileNames: "zjsearch.min.js",
        chunkFileNames: "chunk/[hash].min.js",
        assetFileNames: (asset) => {
          const [name] = asset.names;
          if (name?.endsWith(".css")) {
            return "zjsearch.min[extname]";
          }
          return "assets/[name][extname]";
        },
      },
    },
  },

  plugins: [tailwindcss(), plgAssets(PATH)],
} satisfies import("vite").UserConfig;
