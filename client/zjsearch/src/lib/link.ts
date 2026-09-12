// SPDX-License-Identifier: AGPL-3.0-or-later

/** target/rel attributes for outbound links, following the
    "open results in new tabs" preference (Shell nav, results, infobox
    urls, answers' sources — everywhere an external link is rendered). */
export function newTabLinkProps(onNewTab: boolean | undefined): { target?: string; rel: string } {
  return onNewTab ? { target: "_blank", rel: "noopener noreferrer" } : { rel: "noreferrer" };
}
