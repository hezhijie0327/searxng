# AGENTS.md

Guidance for AI agents working in this repository.

## Repository

Fork of [SearXNG](https://github.com/searxng/searxng) (metasearch engine, Python/Flask + Jinja2).
Current working branch: `skin`. The purpose of this fork is the custom theme
**zjsearch** — a from-scratch React + TypeScript UI — alongside the upstream
`simple` theme. Python changes are the exception, not the rule: only make them
when the user explicitly asks (e.g. the structured `data` payloads that
special-query answers carry for the theme — see
`searx/result_types/answer.py` and the hash/self-info/time-zone plugins plus
the random/statistics answerers).

Key directories:

- `searx/` — SearXNG core (webapp.py, search, engines). Avoid editing unless
  the user directs it; templates under `searx/templates/zjsearch/` are fair game.
- `searx/templates/zjsearch/` — zjsearch theme templates ("data shells").
- `searx/templates/zjsearch/data/macros.html` — the server → client data contract.
- `client/zjsearch/` — React 19 + TS + Vite 8 + Tailwind v4 workspace for zjsearch.
- `client/simple/`, `searx/templates/simple/` — upstream theme, do not refactor.
- `searx/static/themes/simple/` — built assets of the upstream theme (committed
  to git, like upstream). `searx/static/themes/zjsearch/` is **git-ignored**
  (`.gitignore`) — never commit zjsearch build output.
- `utils/lib_sxng_themes.sh`, `utils/lib_sxng_vite.sh` — make targets for themes.

## Commands

```sh
make themes.zjsearch        # npm install + vite build -> searx/static/themes/zjsearch
make themes.zjsearch.lint   # biome check + tsc --noEmit (run inside client/zjsearch)
make themes.zjsearch.dev    # vite dev server (HMR), proxies API calls to :8888
make run                    # dev instance on http://127.0.0.1:8888 (granian, reloads ./searx)
```

- Local instance for theme work (default_theme: zjsearch, all search formats on):
  `SEARXNG_SETTINGS_PATH=$PWD/client/zjsearch/dev-settings.yml ./manage webapp.run`
- First setup: `./manage pyenv.install` (Python venv in `./local/py3`).
- Theme changes require `make themes.zjsearch`; the browser caches assets for 30 s
  (WhiteNoise), reload twice or wait after rebuilding.

## zjsearch architecture (Page-Data pattern)

The server renders **no UI**. Every zjsearch Jinja template is a thin shell that
serializes the render context into `<script id="page-data" type="application/json">`
and boots `zjsearch.min.js`; React renders 100% of the interface.

- The data contract lives in `searx/templates/zjsearch/data/macros.html` and is
  mirrored by TS types in `client/zjsearch/src/lib/types.ts`. **Keep both in sync.**
- Macros apply server filters during serialization: `image_proxify`, `favicon_url`,
  `get_pretty_url`, query highlighting (`title_html`/`content_html` are escaped
  HTML — render with `dangerouslySetInnerHTML`).
- Client-side navigation (`src/lib/router.tsx`) fetches the same URLs and extracts
  the embedded page-data JSON from the HTML response; on network failure it falls
  back to a full page load.
- Client settings come from the base64 `client_settings` attribute on the module
  script tag (`get_client_settings()` in webapp.py). Note: its `theme_static_path`
  is hardcoded to the simple theme — zjsearch uses its own `THEME_STATIC` constant.
- i18n is theme-owned: `client/zjsearch/src/lib/i18n.ts` holds the whole UI
  string catalog (English sources + Simplified Chinese; every other locale
  falls back to English). `globals.strings` is gone from the page-data
  contract. Add new keys to BOTH maps and render via `useT()` / `t("key")`.
- About/Stats/Preferences open as slide-in drawers (`src/lib/overlay.tsx`); the
  panel fetches page-data and renders the same page components with
  `embedded`/`hideTopNav` props. Internal links inside a panel are browsed within
  the panel (click-capture in overlay.tsx).

## Conventions & gotchas

- Jinja macros need `with context` imports to see render variables; Jinja macro
  output is Markup (never escaped), and `|tojson` is the only safe way to embed
  dynamic values — hand-built JSON must place `, ` separators **between** items,
  never before a closing brace.
- Theme style light/dark uses the `simple_style` cookie; `black` is OLED black
  (html gets both `dark` and `black` classes). Auto = no cookie + system setting.
  The html class is set by an inline script in `base.html` and mirrored by
  `applyThemeStyle()` in `client/zjsearch/src/lib/theme.ts` — keep both in sync.
- `POST /preferences` form semantics (upstream `Preferences.parse_form`):
  absent booleans are false; absent `category_*` clears the category
  selection; **`engine_<name>__<category>` and `plugin_<id>` are REVERSED —
  a posted key marks that engine/plugin as *disabled***, and every omitted
  key is re-enabled (a save must always post the complete disabled set);
  other absent key/value settings are left unchanged. The preferences UI
  auto-saves (debounced) with these semantics — no save button, and the
  initial mount must NOT post (it would flip engines/plugins).
- Results hotkeys (default / vim, see `src/features/hotkeys.ts`) must not fire
  while focus is in text inputs; hash-only changes (`#image-viewer`) are ignored
  by the router's popstate handler.
- Bangs: category bangs (`!movies`), engine bangs (`!imdb`, one per engine
  `shortcut`) and external DDG bangs (`!!w`, redirect off-site — the SPA
  fetch fails cross-origin and falls back to a full page load, which is the
  desired behaviour).  Engine bangs run with selected category `"none"`, so
  ResultsPage derives the presentation category from the results' common
  category (`bangCategory`) — that is how `!imdb` lands on the movies
  PosterGrid.  Movies = tmdb/imdb/moviepilot/rottentomatoes/senscritique;
  tmdb is disabled upstream, dev-settings.yml enables it.  Dictionary bangs
  (`!dictionaries` / `!define`) render DictionaryCard word entries; wordnik
  definitions additionally arrive as a translations answer.
- Respect `prefers-reduced-motion`; RTL uses Tailwind logical properties (`ps-`,
  `me-`, `start-`, `end-`) against a single stylesheet.
- Text result cards keep fixed height slots so every card in a list is the
  same height: pretty URL 1 line, title `line-clamp-1`, snippet capped at
  `line-clamp-2` (never reserve empty lines below short snippets — the gap
  reads as broken spacing on pages with 1-line content, e.g. IT), engines
  row capped at 3 pills + "+N". Cards grow only for real content extras
  (publishedDate meta row, thumbnails, embedded media) — do not reserve
  empty slots for those. All text cards share the margin language
  `mt-1` (title, meta) / `mt-1.5` (snippet, tags) / `mt-2` (engines row);
  PackageCard follows it too — version/license live in the meta row, no
  redundant package_name, secondary links fold into the engines row as
  muted chips (`EnginesLine`'s `leading` slot).
- The empty `searx/templates/<name>/` directory alone registers a theme in the
  UI — never leave a half-created theme dir behind.
- Stacking contexts: entrance animations (`animate-fade-up`, fill-mode `both`)
  leave a residual `transform` on their wrapper, which makes every animated
  sibling a stacking context — a `z-30` dropdown inside one of them loses
  against DOM-later siblings (this once let the category tabs and the hotkeys
  hint paint over the homepage autocomplete). Wrappers that contain an overlay
  (autocomplete dropdown, menus) need an explicit raised level such as
  `relative z-10`.

## zjsearch UI design system

A consistent control/typography language is enforced across all pages —
reuse these tokens instead of inventing sizes. The catalog lives in
`client/zjsearch/src/lib/i18n.ts`; strings are looked up by key with
`t("key")`.

Type scale — one size per text role:

- 12px `text-xs`: meta/captions — engine chips, pretty URLs, answers meta,
  mono blocks (URL/hash), footer.
- 13px `text-[13px]`: interactive controls & compact descriptions — category
  tabs, dropdown triggers, pills, help dialog copy, sidebar suggestions,
  preference section tabs.
- 14px `text-sm`: body text and settings row titles.
- 16px `text-base`: result titles (list/news/product/video grids all share
  the `Title`/h3 token) and search inputs.
- 20px `text-xl`: infobox title; 24px `text-2xl`: page headings.
- Brand marks: hero `text-6xl/7xl`, header `text-xl`, both `font-extrabold`.
- Thumbnail corner badges (duration, image count): 11px `font-medium`.

Weights: `font-extrabold` brand only, `font-semibold` headings,
`font-medium` emphasis/selected states; body stays regular.

Controls:

- Circular ghost icon buttons: 36px (`size-9`) with 18px icons
  (`size-[18px]`) — header actions, drawer/help closes, search clear. The
  search submit is the accent-filled circle, also 36px. BackToTop is the
  one floating exception (40px).
- Tab-style buttons (category tabs, filter triggers, preference section
  tabs): `px-4 py-2 text-[13px]`, leading icon 14px.
- Pills/chips (choices, enable/disable, suggestions): `px-3 py-1.5
  text-[13px]` rounded-full; category chips carry `CategoryIcon`; selection
  = `border-accent-strong bg-accent-soft font-medium text-accent`.
- Boxed form selects (preferences): `h-9 text-sm`.
- Category selection uses the tab language (icon + label, selected =
  accent text + amber underline) everywhere — results-page tabs, hero
  grid, preferences default-categories and engine tabs all share
  `CategoryTab`/`CategoryTabs` styling.  Other choices (options, toggles)
  keep the bordered chip language.

Instant answers (Answers.tsx) are tiered:

- Answers without a source url (calculator, time, ip, hash, random) render
  uncarded in the results column — gray lead-in expression, value at 4xl,
  `border-b` divider (Google-style).
- Answers with a source url (definitions) and rich widgets (weather,
  translations) keep the accent card.
- The sidebar hosts knowledge (infobox) and diagnostics only — never
  answers.

Category-specific result presentations (single-category intent pages, see
`ResultsPage.tsx` `is*Page` flags) — each category gets the layout that fits
its content, all sharing one visual language:

- images → masonry `ImageGrid`; videos → `VideoGrid`; music → `MusicGrid`;
  files (torrents) → `FilesGrid`; science → scholarly `PaperCard` list;
  products → `ProductGrid`. Bang-limited searches route the same way via
  `only_template` (`paper`, `torrent`).
- Media grids share one tile anatomy: square/16:9 rounded tile, corner
  badges bottom (duration / filesize bottom-right, favicon bottom-left),
  title + one compact meta row below, cells carry `data-hotkey-index` and
  the `selected` ring so results hotkeys walk grids like lists.
- Playable media uses a centered play button on the tile (hover-revealed
  emphasis); while playing, a close button sits top-right. Music tiles with
  a raw `audio_src` swap to a custom mini player (dimmed cover, big
  play/pause, seek bar) and fall back to the `iframe_src` embed on stream
  error; videos play their embed inside the tile.
- FilesGrid has no cover art: the tile shows a type icon + detected file
  extension, the filesize takes the badge slot, seed/leech health reads as
  colored ↑↓ counts, and the magnet link is an accent circle button.
- Mixed searches render one **collapsible block per original search
  category** (`collectBlocks` / `blockKeyOf` in `ResultsPage`): general,
  images, videos, news, map, music, it, science, files, social media,
  other — pure relevance order inside each block, in tab order by
  default.  Blocks are titled with the bare category name (综合 / 图片 /
  ... via `category_labels`), never with a 结果 suffix.  Each block
  header (category icon + label + count + chevron) toggles collapse —
  folding is the quick-locate mechanism and it works on mobile.  Blocks
  default to expanded; do not reintroduce compact strip previews or
  drag-reorder handles for them (both were tried and removed).  Every block renders the same full presentation as its
  single-category page (`ImageGrid` masonry, `VideoGrid`/`MusicGrid`/
  `FilesGrid`/`PackageGrid` full grids) — never a stripped-down preview.
  Grid cells take `indexOffset` so hotkey indices stay page-global.
  Single-category intent pages (it, ...) keep a plain
  relevance-ordered list instead — extracting a type into a block there
  would break the relevance order (see the `singleCategory` gate in
  `ResultsPage`).  Infinite scroll appends results into their matching
  block; loading pauses while the general block is collapsed.

Results right rail (desktop): the infobox scrolls inside its own area
(`min-h-0 flex-1 overflow-y-auto`); hide the rail area entirely when its
content is empty (e.g. "test"-style searches with no infobox) so blank
space never pushes content down. Diagnostics (`DebugPanels`) live in the
results meta line on both desktop and mobile: the line reads 「找到 N 条
相关结果 · 耗时 X.X 秒 ▾」 and clicking it expands a single engine-timing
table — unresponsive engines share the same grid (red error label + empty
bar, seconds column aligned); the panel starts expanded when there are
zero results. Suggestions render as a single-row chip strip
under the results meta line (`SuggestionsBox`, all breakpoints): chips
are single-line truncated, the row pages via ‹ › ghost arrows that stay
persistent (disabled at the ends and when the row fits, so flipping state
never shifts the chips; uncapped — paging handles any count, honors
`prefers-reduced-motion`); the right rail never hosts suggestions.

Query-term highlighting (`.highlight` in global.css) is a tinted
background only — color marks the term, no bold.

## zjsearch performance notes

- Every content `<img>` is `loading="lazy" decoding="async"` inside an
  aspect-ratio container (no CLS); the first four result thumbnails are
  `loading="eager" fetchPriority="high"` (LCP).
- Route-level code splitting: Preferences/Stats/Info pages load through
  `src/pages/lazyPages.ts` (`React.lazy` + `Suspense` skeleton fallbacks in
  app.tsx and overlay.tsx); OpenLayers is dynamically imported only when a
  map result expands. Keep heavy features out of the eager graph.
- No webfonts (system font stack) and no third-party scripts; icons are
  inline SVG (`src/components/icons.tsx`), never an icon font.
- Drawer/lightbox overlays render conditionally (zero cost when closed).

## Windows (Git Bash) development notes

`./manage` and `make` assume a POSIX host and **do not work on Windows**:
`utils/lib.sh` sources `/etc/os-release`, `manage` needs a `python3` command
and a POSIX venv layout (`local/py3/bin/python`), and `make` is absent from
Git Bash. To run the dev instance on Windows anyway (all outside the repo, no
Python edits — the repo policy forbids them):

- Create the venv manually: `python -m venv local/py3` (Windows layout:
  `local/py3/Scripts/`), then
  `local/py3/Scripts/python -m pip install -r requirements.txt -r requirements-dev.txt`.
- `searx/valkeydb.py` imports the POSIX-only `pwd` module at top level and
  crashes on import. Put a tiny `pwd` stub outside the repo on `PYTHONPATH`.
- Windows path separators break theme asset URLs:
  `webutils.get_static_file_list()` returns `themes\zjsearch\...` while
  `webapp.custom_url_for` compares with forward slashes, so `/static/zjsearch.min.js`
  is served unmapped and 404s (works fine on POSIX). Workaround: hardlink the
  built assets into `searx/static/` with `cmd //c "mklink /H zjsearch.min.js
  themes\zjsearch\zjsearch.min.js"` (same for `.css`). These links are
  untracked, must not be committed, and go stale after every rebuild (vite
  replaces the target file) — delete and recreate them.
- Start the app directly, mirroring `manage`'s `webapp.run` env vars:
  `SEARXNG_SETTINGS_PATH=<settings.yml> GRANIAN_INTERFACE=wsgi
  GRANIAN_HOST=127.0.0.1 GRANIAN_PORT=8888 local/py3/Scripts/granian
  searx.webapp:app` (needs the `granian[pname,reload]` extra for the
  `GRANIAN_PROCESS_NAME`/reload vars; they can simply be omitted).
- Granian workers inherit the listening socket. Killing the shell wrapper (or
  a task manager's "stop") can leave an orphan worker bound to :8888; a second
  instance can then bind the same port too (SO_REUSEADDR) and requests race
  between an old and a new server — the classic symptom is "rebuilt assets
  but the page serves stale ones". Before starting an instance, check
  `netstat -ano | grep :8888` and `taskkill //PID <pid> //F` every listener.

## Docs worth reading first

- `client/zjsearch/README.rst` — theme architecture and workflow.
- `docs/dev/templates.rst` — result field reference (the upstream render contract).
- `docs/dev/plugins/` — server plugin registry used by the preferences UI.
