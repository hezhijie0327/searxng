// SPDX-License-Identifier: AGPL-3.0-or-later

import { useEffect, useState } from "react";
import { useT } from "../lib/i18n.ts";
import { ArrowUpIcon } from "./icons.tsx";

export function BackToTop() {
  const t = useT();
  const [visible, setVisible] = useState(false);
  useEffect(() => {
    const onScroll = () => {
      setVisible(window.scrollY > 400);
    };
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => {
      window.removeEventListener("scroll", onScroll);
    };
  }, []);
  if (!visible) {
    return null;
  }
  return (
    <button
      aria-label={t("back_to_top")}
      className="fixed bottom-6 right-6 z-40 grid size-11 place-items-center rounded-full border border-line bg-surface text-ink-2 shadow-pop transition-colors hover:text-accent animate-fade-in"
      onClick={() => {
        window.scrollTo({ top: 0, behavior: "smooth" });
      }}
      type="button"
    >
      <ArrowUpIcon className="size-5" />
    </button>
  );
}
