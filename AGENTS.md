# AGENTS.md

Guidance for AI agents working in this repository.

## Repository

Fork of [SearXNG](https://github.com/searxng/searxng) (metasearch engine, Python/Flask + Jinja2).
Current working branch: `skin`. The purpose of this fork is the custom theme
**zjsearch** — a from-scratch React + TypeScript UI — alongside the upstream
`simple` theme, **without modifying any Python code**.

Key directories:

- `searx/` — SearXNG core (webapp.py, search, engines, templates). Avoid editing.
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
- i18n: all UI strings are translated server-side into `globals.strings` (see the
  `_strings()` macro). Use msgids that match `searx/translations/*.po` exactly
  (e.g. lowercase `_('auto')`, not `_('Auto')`) or translations silently fall back.
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
- Respect `prefers-reduced-motion`; RTL uses Tailwind logical properties (`ps-`,
  `me-`, `start-`, `end-`) against a single stylesheet.
- The empty `searx/templates/<name>/` directory alone registers a theme in the
  UI — never leave a half-created theme dir behind.
- Stacking contexts: entrance animations (`animate-fade-up`, fill-mode `both`)
  leave a residual `transform` on their wrapper, which makes every animated
  sibling a stacking context — a `z-30` dropdown inside one of them loses
  against DOM-later siblings (this once let the category tabs and the hotkeys
  hint paint over the homepage autocomplete). Wrappers that contain an overlay
  (autocomplete dropdown, menus) need an explicit raised level such as
  `relative z-10`.

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
