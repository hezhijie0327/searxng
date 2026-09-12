// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * Data contract between the SearXNG server (searx/templates/zjsearch) and this
 * React application.  Every type mirrors a JSON structure emitted by the
 * Jinja serialization macros in searx/templates/zjsearch/data/macros.html.
 */

// ------------------------------------------------------------------ globals

export interface LocaleInfo {
  tag: string;
  name: string;
  country: string;
  english: string;
  flag: string;
}

export interface CustomLink {
  title: string;
  url: string;
}

export interface GlobalData {
  page: "index" | "results" | "preferences" | "stats" | "info" | "404";
  instance_name: string;
  version: string;
  git_url: string;
  issue_url: string;
  donation_url: string;
  enable_metrics: boolean;
  public_instances_url: string;
  privacypolicy_url: string;
  contact_url: string;
  custom_links: CustomLink[];
  search_formats: string[];
  categories_as_tabs: string[];
  categories: string[];
  default_category: string;
  category_labels: Record<string, string>;
  locales: LocaleInfo[];
  language: string;
  safesearch: 0 | 1 | 2;
  method: "GET" | "POST";
  theme: string;
  theme_style: "auto" | "light" | "dark" | "black";
  search_on_category_select: boolean;
  results_on_new_tab: boolean;
  url_formatting: "pretty" | "full" | "host";
  favicon_resolver: string;
  doi_resolver: string;
  cache_url: string;
  locale: string;
  about_url: string;
  search_syntax_url: string;
  rtl: boolean;
}

// ------------------------------------------------------------------ results

export interface ResultItem {
  template: string;
  url: string;
  title_html: string;
  title_text: string;
  content_html: string;
  content_text: string;
  engines: string[];
  category: string;
  priority: string;
  /** relevance score from the backend: sum of engine weight / position */
  score?: number;

  netloc?: string;
  favicon?: string;
  pretty_url?: string[];
  published_date?: string;
  length_display?: string;
  length_seconds?: number;
  views?: string;
  author?: string;
  metadata?: string;
  img_src?: string;
  thumbnail?: string;
  iframe_src?: string;
  audio_src?: string;
  open_group?: boolean;
  close_group?: boolean;

  // images.html
  thumbnail_src?: string;
  resolution?: string;
  img_format?: string;
  source?: string;
  filesize?: string;
  formats?: Array<{ url: string; label: string }>;

  // torrent.html
  magnetlink?: string;
  torrentfile?: string;
  seed?: number;
  leech?: number;
  files?: number;

  // map.html
  address?: Partial<{
    name: string;
    road: string;
    house_number: string;
    postcode: string;
    locality: string;
    country: string;
    country_code: string;
  }>;
  map_links?: Array<{ label: string; url: string }>;
  data?: Array<{ label: string; value: string }>;
  boundingbox?: number[];
  longitude?: string;
  latitude?: string;
  geojson?: unknown;

  // paper.html
  comments?: string;
  tags?: string[];
  paper_type?: string;
  authors?: string[];
  editor?: string;
  publisher?: string;
  journal?: string;
  volume?: string;
  number?: string;
  pages?: string;
  doi?: string;
  issn?: string[];
  isbn?: string[];
  pdf_url?: string;
  html_url?: string;

  // packages.html
  package_name?: string;
  version?: string;
  maintainer?: string;
  popularity?: string;
  license_name?: string;
  license_url?: string;
  homepage?: string;
  source_code_url?: string;
  project_links?: Record<string, string>;

  // code.html
  repository?: string;
  filename?: string;
  code_html?: string;

  // file.html
  abstract_html?: string;
  size?: string;
  time?: string;
  mimetype?: string;
  embedded?: string;
  mtype?: string;
  subtype?: string;

  // keyvalue.html
  caption?: string;
  key_title?: string;
  value_title?: string;
  kvmap?: Record<string, unknown>;

  // products.html
  price?: string;
  shipping?: string;
  source_country?: string;
}

export interface SuggestionItem {
  q: string;
  title: string;
}

export interface TranslationItem {
  text: string;
  transliteration: string;
  examples: string[];
  definitions: string[];
  synonyms: string[];
}

export interface WeatherItem {
  summary: string;
  symbol: string;
  temperature: string;
  datetime_display?: string;
  feels_like?: string;
  wind?: string;
  wind_speed?: string;
  pressure?: string;
  humidity?: string;
}

export type AnswerData =
  | {
      template: "answer/legacy.html";
      answer: string;
      url: string;
    }
  | {
      template: "answer/translations.html";
      engine: string;
      url: string;
      translations: TranslationItem[];
    }
  | {
      template: "answer/weather.html";
      service: string;
      url: string;
      current: WeatherItem;
      forecasts: WeatherItem[];
    };

export interface InfoboxData {
  title: string;
  content_html: string;
  img_src: string;
  attributes?: Array<{ label: string; value: string; image_src: string; image_alt: string }>;
  urls?: Array<{ title: string; url: string }>;
  related_topics?: Array<{ name: string; suggestions: string[] }>;
}

export interface SearchPageData {
  globals: GlobalData;
  q: string;
  selected_categories: string[];
  pageno: number;
  time_range: string;
  suggestions: SuggestionItem[];
  corrections: SuggestionItem[];
  answers: AnswerData[];
  infoboxes: InfoboxData[];
  results: ResultItem[];
  engine_data: Record<string, Record<string, string>>;
  paging: boolean;
  unresponsive_engines: Array<[string, string]>;
  timings: Array<{ name: string; time: number }>;
  max_response_time: number | null;
  timeout_limit: string;
  current_language: string;
  search_language: string;
  only_template: string;
}

// -------------------------------------------------------------- preferences

export interface EngineEntry {
  name: string;
  shortcut: string;
  categories: string[];
  language: string;
  enable_http: boolean;
  weight: number;
  timeout: number;
  website: string;
  wikidata_id: string;
  disabled: boolean;
  stats_time: number | null;
  stats_rate80: number | null;
  stats_rate95: number | null;
  stats_result_count: number;
  warn_timeout: boolean;
  reliability: number | null;
  errors: string[];
  supports_safesearch: boolean;
  supports_time_range: boolean;
}

export interface EngineGroup {
  group: string;
  group_bang: string;
  engines: EngineEntry[];
}

export interface EngineTab {
  category: string;
  label: string;
  is_default: boolean;
  groups: EngineGroup[];
}

export interface PluginInfo {
  id: string;
  name: string;
  description: string;
  section: string;
  enabled: boolean;
  keywords: string[];
  examples: string[];
  plugin: boolean;
}

export interface PreferencesPageData {
  globals: GlobalData;
  current_locale: string;
  locales: Record<string, string>;
  kv: {
    categories: string[];
    language: string;
    locale: string;
    autocomplete: string;
    favicon_resolver: string;
    safesearch: 0 | 1 | 2;
    theme: string;
    simple_style: string;
    center_alignment: boolean;
    results_on_new_tab: boolean;
    search_on_category_select: boolean;
    hotkeys: string;
    url_formatting: string;
    method: "GET" | "POST";
    image_proxy: boolean;
    query_in_title: boolean;
    doi_resolver: string;
    tokens: string;
  };
  autocomplete_backends: string[];
  favicon_resolver_names: string[];
  themes: string[];
  doi_resolvers: Record<string, string>;
  locked_preferences: string[];
  preferences_url_params: string;
  cookies: Array<{ name: string; value: string }>;
  answerers: Array<{ keywords: string[]; name: string; description: string; examples: string[]; plugin: boolean }>;
  plugins: PluginInfo[];
  engine_tabs: EngineTab[];
}

// -------------------------------------------------------------------- stats

export interface EngineStat {
  name: string;
  score: number | null;
  score_per_result: number | null;
  result_count: number | null;
  total: number | null;
  http: number | null;
  processing: number | null;
  total_p80: number | null;
  http_p80: number | null;
  processing_p80: number | null;
  total_p95: number | null;
  http_p95: number | null;
  processing_p95: number | null;
  reliability: number | null;
}

export interface EngineError {
  secondary: boolean;
  exception_classname: string;
  log_message: string;
  percentage: number;
  log_parameters: string[];
  filename: string;
  line_no: number;
  function: string;
  code: string;
}

export interface StatsPageData {
  globals: GlobalData;
  sort_order: string;
  selected_engine_name: string;
  max_result_count: number;
  max_time: number;
  engines: EngineStat[];
  errors: EngineError[];
}

// --------------------------------------------------------------------- info

export interface InfoPageData {
  globals: GlobalData;
  active_pagename: string;
  active_title: string;
  active_html: string;
  pages: Array<{ pagename: string; locale: string; title: string }>;
}

// -------------------------------------------------------------- page union

export interface BasicPageData {
  globals: GlobalData;
}

export type AnyPageData = SearchPageData | PreferencesPageData | StatsPageData | InfoPageData | BasicPageData;

export function isSearchPageData(data: AnyPageData): data is SearchPageData {
  return data.globals.page === "results";
}

export function isPreferencesPageData(data: AnyPageData): data is PreferencesPageData {
  return data.globals.page === "preferences";
}

export function isStatsPageData(data: AnyPageData): data is StatsPageData {
  return data.globals.page === "stats";
}

export function isInfoPageData(data: AnyPageData): data is InfoPageData {
  return data.globals.page === "info";
}
