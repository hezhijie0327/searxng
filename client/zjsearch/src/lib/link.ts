// SPDX-License-Identifier: Apache-2.0 WITH Commons-Clause-1.0

/** target/rel attributes for outbound links, following the
    "open results in new tabs" preference (Shell nav, results, infobox
    urls, answers' sources — everywhere an external link is rendered). */
export function newTabLinkProps(onNewTab: boolean | undefined): { target?: string; rel: string } {
  return onNewTab ? { target: "_blank", rel: "noopener noreferrer" } : { rel: "noreferrer" };
}
