// SPDX-License-Identifier: Apache-2.0 WITH Commons-Clause-1.0

/** The ZJSearch brand mark (logo). */

export function BrandMark({ className }: { className?: string }) {
  return (
    <svg aria-hidden="true" className={className} focusable="false" viewBox="0 0 100 100">
      <defs>
        <linearGradient id="zjs-brand-gradient" x1="0" x2="1" y1="0" y2="1">
          <stop offset="0" stopColor="#ffd76b" />
          <stop offset="1" stopColor="#f0b429" />
        </linearGradient>
      </defs>
      <rect fill="#211f1c" height="100" rx="22" width="100" />
      <circle cx="43" cy="43" fill="url(#zjs-brand-gradient)" r="24" stroke="#f5f2ea" strokeWidth="9" />
      <path d="M61 61 L82 82" stroke="#f5f2ea" strokeLinecap="round" strokeWidth="11" />
      <path
        d="M33 36 a13 13 0 0 1 10 -6"
        fill="none"
        opacity="0.85"
        stroke="#ffffff"
        strokeLinecap="round"
        strokeWidth="5"
      />
    </svg>
  );
}
