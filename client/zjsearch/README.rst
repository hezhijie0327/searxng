=====
zjsearch
=====

A from-scratch React + TypeScript UI theme for SearXNG, named **ZJSearch**.

Unlike the bundled ``simple`` theme (server-rendered HTML progressively
enhanced with TypeScript), zjsearch renders **100% of the UI in React**.
The SearXNG server keeps rendering every page, but the theme's Jinja
templates are thin *data shells*: they serialize the full render context
into a JSON payload (``<script id="page-data" type="application/json">``)
and boot the React bundle. No Python files are modified — the theme is
purely additive and survives upstream upgrades.

Architecture
============

- **Page-Data pattern** — every view template
  (``searx/templates/zjsearch/*.html``) emits the server context as JSON
  via macros in ``data/macros.html``. Server-side filters
  (``image_proxify``, ``favicon_url``, ``get_pretty_url``, query
  highlighting, ``engine_data`` for pagination) are applied during
  serialization, so the React client inherits the complete data pipeline
  without any backend changes.
- **Client-side navigation** — the React router (``src/lib/router.tsx``)
  fetches the same ``/search`` / ``/preferences`` / … URLs, extracts the
  embedded ``page-data`` JSON from the HTML response and swaps the page
  via ``pushState``. A progress bar + skeletons cover the wait.
- **i18n** — the UI string catalog is translated server-side with
  ``_()`` and embedded in ``globals.strings``; the client only looks keys
  up (``src/lib/i18n.ts``).
- **Preferences** — the settings UI builds a ``FormData`` that mirrors the
  upstream ``parse_form`` semantics (absent booleans are false, checked
  ``engine_<name>__<category>`` boxes mean *allowed*, ``plugin_<id>``
  means *enabled*) and POSTs it to ``/preferences``.
- **Stack** — React 19, TypeScript, Vite (rolldown), Tailwind CSS v4
  (warm Kagi-inspired palette, light/dark/auto via the ``simple_style``
  cookie), Biome for lint/format.

Layout of this workspace
========================

.. code:: text

   client/zjsearch/
   ├── dev-settings.yml     # local instance settings (default_theme: zjsearch)
   ├── vite.config.ts       # build -> searx/static/themes/zjsearch, dev proxy
   ├── tools/assets.ts      # rasterizes brand SVGs into favicons/PWA icons
   └── src/
       ├── main.tsx         # boot: parse page-data + client_settings, mount
       ├── app.tsx          # page switch (index/results/preferences/stats/…)
       ├── styles/global.css  # Tailwind v4 theme tokens, light/dark, motion
       ├── lib/             # router, page-data, settings, i18n, format
       ├── components/      # shell, search box, filters, result cards, …
       └── pages/           # Index / Results / Preferences / Stats / Info

Build & development
===================

.. code:: sh

   make themes.zjsearch        # npm install + vite build (into searx/static/themes/zjsearch)
   make themes.zjsearch.lint   # biome check + tsc --noEmit
   make themes.zjsearch.dev    # vite dev server (HMR), proxies API calls to
                               # a local instance; start the instance with:
                               # SEARXNG_SETTINGS_PATH=client/zjsearch/dev-settings.yml ./manage webapp.run

The dev server proxies ``/search``, ``/autocompleter``, ``/preferences``,
``/image_proxy``, … to ``http://127.0.0.1:8888`` (override with
``ZJSEARCH_BACKEND``), so the UI can be developed against a live backend.

Selecting the theme
===================

The theme becomes selectable by merely existing (a template directory is a
theme). Set it as default in your ``settings.yml``:

.. code:: yaml

   ui:
     default_theme: zjsearch

or pick *ZJSearch → Theme* in the preferences UI.

Feature checklist (parity with ``simple``)
==========================================

- Search box with debounced autocompleter (keyboard navigation), clear
  button, ``dir="auto"`` queries
- Category tabs (single select / shift-click multi select, honoring
  ``search_on_category_select``), language / time range / safesearch
  filters that re-search on change
- All 11 result layouts (default, images, videos, torrent, map, paper,
  packages, code, file, keyvalue, products) + answers (legacy,
  translations, weather), corrections, suggestions, infobox with related
  topics, engine messages with response-time bars, permalink box (POST
  mode), download-results links
- Image results: masonry grid for image-only pages, lightbox with
  keyboard/swipe navigation and ``#image-viewer`` back-button dismissal,
  proxified thumbnails with error fallback
- Numbered pagination window + plugin-gated infinite scroll,
  back-to-top
- Preferences: all six tabs, engine table with reliability/response
  stats and lazy engine descriptions, plugins, tokens, cookie list,
  preferences share/copy/restore, preview banner, reset defaults
- Stats: sortable table with bar charts + per-engine error logs
- Info/about pages, 404, OpenSearch XML, RSS (rendered server side),
  PWA manifest and icons
- Light/dark/auto theme with quick toggle (``simple_style`` cookie),
  RTL support via logical properties, ``prefers-reduced-motion`` support
- Client plugins gated by ``client_settings.plugins``: infinite scroll,
  calculator*, map results with lazy-loaded OpenLayers embed

\* calculator render is handled by the server answerer; client plugin
parity is on the roadmap.
