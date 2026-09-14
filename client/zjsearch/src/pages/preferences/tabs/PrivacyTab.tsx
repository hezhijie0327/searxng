// SPDX-License-Identifier: Apache-2.0 WITH Commons-Clause-1.0

import { ArrowLeftRight, Image, Tag } from "lucide-react";
import { useT } from "@/lib/i18n.ts";
import type { PreferencesPageData } from "@/lib/types.ts";
import { Card, PluginRow, SettingRow, Switch } from "@/pages/preferences/parts.tsx";
import type { PreferencesForm } from "@/pages/preferences/usePreferencesForm.ts";

export function PrivacyTab({
  data,
  form,
  locked,
}: {
  data: PreferencesPageData;
  form: PreferencesForm;
  locked: Set<string>;
}) {
  const t = useT();
  return (
    <Card>
      {!locked.has("method") ? (
        <SettingRow
          description={t("change_forms_submit")}
          icon={<ArrowLeftRight className="size-4.5" />}
          title={t("http_method")}
        >
          <div className="inline-flex rounded-xl border border-line bg-surface p-0.5">
            {(["POST", "GET"] as const).map((value) => (
              <button
                aria-pressed={form.method === value}
                className={`rounded-lg px-4 py-1.5 text-[13px] transition-colors ${
                  form.method === value
                    ? "bg-accent-strong font-medium text-accent-contrast"
                    : "text-ink-2 hover:text-ink"
                }`}
                key={value}
                onClick={() => {
                  form.setMethod(value);
                }}
                type="button"
              >
                {value}
              </button>
            ))}
          </div>
        </SettingRow>
      ) : null}
      {!locked.has("image_proxy") ? (
        <SettingRow description={t("proxy_images")} icon={<Image className="size-4.5" />} title={t("image_proxy")}>
          <Switch checked={form.imageProxy} label={t("image_proxy")} onChange={form.setImageProxy} />
        </SettingRow>
      ) : null}
      {!locked.has("query_in_title") ? (
        <SettingRow
          description={t("query_in_title_desc")}
          icon={<Tag className="size-4.5" />}
          title={t("query_in_title")}
        >
          <Switch checked={form.queryInTitle} label={t("query_in_title")} onChange={form.setQueryInTitle} />
        </SettingRow>
      ) : null}
      {data.plugins
        .filter((plugin) => plugin.section === "privacy")
        .map((plugin) => (
          <PluginRow
            enabled={form.plugins[plugin.id] ?? false}
            key={plugin.id}
            onChange={(checked) => {
              form.setPluginEnabled(plugin.id, checked);
            }}
            plugin={plugin}
          />
        ))}
    </Card>
  );
}
