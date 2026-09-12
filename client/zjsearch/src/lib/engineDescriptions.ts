// SPDX-License-Identifier: Apache-2.0 WITH Commons-Clause-1.0

/**
 * Lazy engine descriptions (same source as the simple theme):
 * GET /engine_descriptions.json -> { engine: [description, source] }
 */

let cache: Record<string, [string, string]> | null = null;
let inflight: Promise<Record<string, [string, string]>> | null = null;

export function loadEngineDescriptions(): Promise<Record<string, [string, string]>> {
  if (cache) {
    return Promise.resolve(cache);
  }
  if (!inflight) {
    inflight = fetch("engine_descriptions.json")
      .then(async (resp) => {
        cache = resp.ok ? ((await resp.json()) as Record<string, [string, string]>) : {};
        return cache;
      })
      .catch(() => {
        cache = {};
        return cache;
      });
  }
  return inflight;
}
