// SPDX-License-Identifier: Apache-2.0 WITH Commons-Clause-1.0

import { type StringKey, useT } from "@/lib/i18n.ts";
import type { PreferencesPageData } from "@/lib/types.ts";
import { pluginLabels, Switch } from "@/pages/preferences/parts.tsx";
import type { PreferencesForm } from "@/pages/preferences/usePreferencesForm.ts";

/** ZJSearch-side labels for the built-in answerers (server strings stay
    untranslated — same rationale as PLUGIN_I18N in parts.tsx). */
const ANSWERER_I18N: Record<string, { description: StringKey; name: StringKey }> = {
  "Random value generator": { description: "answerer_random_desc", name: "answerer_random" },
  "Statistics functions": { description: "answerer_stats_desc", name: "answerer_stats" },
};

function answererLabels(name: string, description: string, t: ReturnType<typeof useT>) {
  const keys = ANSWERER_I18N[name];
  return keys ? { description: t(keys.description), name: t(keys.name) } : { description, name };
}

export function QueryTab({ data, form }: { data: PreferencesPageData; form: PreferencesForm }) {
  const t = useT();
  return (
    <div className="overflow-x-auto rounded-2xl border border-line bg-surface">
      <table className="w-full min-w-[560px] text-left text-xs">
        <thead className="bg-surface-2 text-ink-3">
          <tr>
            <th className="px-4 py-3 font-medium">{t("allow")}</th>
            <th className="px-4 py-3 font-medium">{t("keywords")}</th>
            <th className="px-4 py-3 font-medium">{t("name")}</th>
            <th className="px-4 py-3 font-medium">{t("description")}</th>
            <th className="px-4 py-3 font-medium">{t("examples")}</th>
          </tr>
        </thead>
        <tbody>
          <tr className="bg-surface-2/60">
            <td className="px-4 py-2.5 font-medium text-ink-2" colSpan={5}>
              {t("instant_answer_modules")}
            </td>
          </tr>
          {data.answerers.map((answerer) => {
            const labels = answererLabels(answerer.name, answerer.description, t);
            return (
              <tr className="border-t border-line" key={answerer.name}>
                <td className="px-4 py-3 text-center text-ink-3">–</td>
                <td className="px-4 py-3">
                  <code className="rounded bg-surface-2 px-1.5 py-0.5">{answerer.keywords.join(", ")}</code>
                </td>
                <td className="px-4 py-2.5 font-medium">{labels.name}</td>
                <td className="px-4 py-2.5 text-ink-2">{labels.description}</td>
                <td className="px-4 py-2.5 text-ink-2">{answerer.examples.join(", ")}</td>
              </tr>
            );
          })}
          <tr className="bg-surface-2/60">
            <td className="px-4 py-2.5 font-medium text-ink-2" colSpan={5}>
              {t("plugins_list")}
            </td>
          </tr>
          {data.plugins
            .filter((plugin) => plugin.section === "query")
            .map((plugin) => {
              const labels = pluginLabels(plugin, t);
              return (
                <tr className="border-t border-line" key={plugin.id}>
                  <td className="px-4 py-3">
                    <Switch
                      checked={form.plugins[plugin.id] ?? false}
                      label={labels.name}
                      onChange={(value) => {
                        form.setPluginEnabled(plugin.id, value);
                      }}
                    />
                  </td>
                  <td className="px-4 py-3">
                    <code className="rounded bg-surface-2 px-1.5 py-0.5">{plugin.keywords.join(", ")}</code>
                  </td>
                  <td className="px-4 py-2.5 font-medium">{labels.name}</td>
                  <td className="px-4 py-2.5 text-ink-2">{labels.description}</td>
                  <td className="px-4 py-2.5 text-ink-2">{plugin.examples.join(", ")}</td>
                </tr>
              );
            })}
        </tbody>
      </table>
    </div>
  );
}
