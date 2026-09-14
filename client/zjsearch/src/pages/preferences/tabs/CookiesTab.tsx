// SPDX-License-Identifier: Apache-2.0 WITH Commons-Clause-1.0

import { Cookie, ExternalLink, Key, Link as LinkIcon, RefreshCw } from "lucide-react";
import { ClickToCopy } from "@/components/CopyButton.tsx";
import { Link } from "@/components/Shell.tsx";
import { useT } from "@/lib/i18n.ts";
import type { PreferencesPageData } from "@/lib/types.ts";
import { Card, SettingRow } from "@/pages/preferences/parts.tsx";
import type { PreferencesForm } from "@/pages/preferences/usePreferencesForm.ts";

export function CookiesTab({ data, form }: { data: PreferencesPageData; form: PreferencesForm }) {
  const t = useT();
  const shareOrigin = window.location.origin;
  return (
    <div className="space-y-4">
      <Card>
        <SettingRow description={t("cookies_list_desc")} icon={<Cookie className="size-4.5" />} title={t("cookies")}>
          <span className="text-xs text-ink-3">
            {data.cookies.length > 0 ? `${data.cookies.length}` : t("no_cookies")}
          </span>
        </SettingRow>
      </Card>
      {data.cookies.length > 0 ? (
        <div className="overflow-hidden rounded-2xl border border-line bg-surface">
          <table className="w-full text-left text-xs">
            <thead className="bg-surface-2 text-ink-3">
              <tr>
                <th className="px-4 py-2 font-medium">{t("cookie_name")}</th>
                <th className="px-4 py-2 font-medium">{t("value")}</th>
              </tr>
            </thead>
            <tbody>
              {data.cookies.map((cookie) => (
                <tr className="border-t border-line" key={cookie.name}>
                  <td className="px-4 py-1.5 font-mono">{cookie.name}</td>
                  <td className="px-4 py-1.5 break-all" title={cookie.value.length > 64 ? cookie.value : undefined}>
                    {cookie.value.length > 64 ? `${cookie.value.slice(0, 64)}…` : cookie.value}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : null}

      <Card>
        <div className="px-4 py-4 sm:px-5">
          <div className="flex items-center justify-between gap-2">
            <h4 className="flex items-center gap-2 text-sm font-semibold text-ink">
              <LinkIcon className="size-4 text-accent" />
              {t("search_url_of_prefs")}
            </h4>
          </div>
          <ClickToCopy className="mt-2" value={`${shareOrigin}/?preferences=${data.preferences_url_params}&q=%s`}>
            <pre
              className="max-h-28 min-w-0 overflow-y-auto rounded-xl bg-surface-2 p-2.5 font-mono text-xs break-all whitespace-pre-wrap text-ink-2"
              dir="ltr"
            >
              {shareOrigin}/?preferences={data.preferences_url_params}&amp;q=%s
            </pre>
          </ClickToCopy>
          <p className="mt-1.5 text-xs text-ink-3">{t("prefs_url_privacy_note")}</p>
        </div>
        <div className="px-4 py-4 sm:px-5">
          <div className="flex items-center justify-between gap-2">
            <h4 className="flex items-center gap-2 text-sm font-semibold text-ink">
              <ExternalLink className="size-4 text-accent" />
              {t("url_to_restore")}
            </h4>
          </div>
          <ClickToCopy className="mt-2" value={`${shareOrigin}/preferences?preferences=${data.preferences_url_params}`}>
            <pre
              className="max-h-28 min-w-0 overflow-y-auto rounded-xl bg-surface-2 p-2.5 font-mono text-xs break-all whitespace-pre-wrap text-ink-2"
              dir="ltr"
            >
              {shareOrigin}/preferences?preferences={data.preferences_url_params}
            </pre>
          </ClickToCopy>
          <p className="mt-1.5 text-xs text-ink-3">{t("url_restore_desc")}</p>
        </div>
        <div className="px-4 py-4 sm:px-5">
          <h4 className="flex items-center gap-2 text-sm font-semibold text-ink">
            <Key className="size-4 text-accent" />
            {t("copy_prefs_hash")}
          </h4>
          <ClickToCopy value={data.preferences_url_params}>
            <pre
              className="max-h-28 min-w-0 overflow-y-auto rounded-xl bg-surface-2 p-2.5 font-mono text-xs break-all whitespace-pre-wrap text-ink-2"
              dir="ltr"
            >
              {data.preferences_url_params}
            </pre>
          </ClickToCopy>
        </div>
        <SettingRow icon={<RefreshCw className="size-4.5" />} stacked title={t("insert_prefs_hash")}>
          <input
            aria-label={t("insert_prefs_hash")}
            className="h-9 w-full rounded-xl border border-line bg-surface px-3 text-sm transition-colors hover:border-ink-3"
            onChange={(event) => {
              form.setPastedHash(event.target.value);
            }}
            placeholder={t("prefs_hash")}
            type="text"
            value={form.pastedHash}
          />
        </SettingRow>
      </Card>
      <Card>
        <div className="flex flex-wrap items-center justify-between gap-3 px-4 py-4 sm:px-5">
          <p className="min-w-0 flex-1 text-xs leading-relaxed text-ink-3">
            {t("settings_in_cookies")}
            <br />
            {t("cookies_convenience")}
          </p>
          <Link
            className="inline-flex shrink-0 items-center gap-1.5 rounded-full border border-line bg-surface px-4 py-2 text-[13px] text-ink-2 transition-colors hover:border-danger hover:text-danger"
            href="/clear_cookies"
          >
            <RefreshCw className="size-4" />
            {t("reset_defaults")}
          </Link>
        </div>
      </Card>
    </div>
  );
}
