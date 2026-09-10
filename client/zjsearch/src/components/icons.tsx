// SPDX-License-Identifier: AGPL-3.0-or-later

/** Inline SVG icon set (stroke style, 24x24 grid) + brand marks. */

import type { ReactNode } from "react";

interface IconProps {
  className?: string;
}

function makeIcon(paths: ReactNode, viewBox = "0 0 24 24") {
  return function Icon({ className }: IconProps) {
    return (
      <svg
        aria-hidden="true"
        className={className}
        fill="none"
        focusable="false"
        stroke="currentColor"
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth="2"
        viewBox={viewBox}
      >
        {paths}
      </svg>
    );
  };
}

export const SearchIcon = makeIcon(
  <>
    <circle cx="11" cy="11" r="8" />
    <path d="m21 21-4.35-4.35" />
  </>,
);

export const CloseIcon = makeIcon(
  <>
    <path d="M18 6 6 18" />
    <path d="m6 6 12 12" />
  </>,
);

export const GlobeIcon = makeIcon(
  <>
    <circle cx="12" cy="12" r="10" />
    <path d="M2 12h20" />
    <path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z" />
  </>,
);

export const ImageIcon = makeIcon(
  <>
    <rect height="18" rx="2" width="18" x="3" y="3" />
    <circle cx="8.5" cy="8.5" r="1.5" />
    <path d="m21 15-5-5L5 21" />
  </>,
);

export const PlayIcon = makeIcon(<polygon points="6 3 20 12 6 21 6 3" />);

export const NewsIcon = makeIcon(
  <>
    <path d="M4 22h16a2 2 0 0 0 2-2V4a2 2 0 0 0-2-2H8a2 2 0 0 0-2 2v16a2 2 0 0 1-4 0V9" />
    <path d="M18 14h-8" />
    <path d="M15 18h-5" />
    <path d="M10 6h8v4h-8V6z" />
  </>,
);

export const LayersIcon = makeIcon(
  <>
    <polygon points="12 2 2 7 12 12 22 7 12 2" />
    <polyline points="2 17 12 22 22 17" />
    <polyline points="2 12 12 17 22 12" />
  </>,
);

export const LocationIcon = makeIcon(
  <>
    <path d="M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0 1 18 0z" />
    <circle cx="12" cy="10" r="3" />
  </>,
);

export const MusicIcon = makeIcon(
  <>
    <path d="M9 18V5l12-2v13" />
    <circle cx="6" cy="18" r="3" />
    <circle cx="18" cy="16" r="3" />
  </>,
);

export const FlaskIcon = makeIcon(
  <>
    <path d="M10 2v7.5L4.5 19a2 2 0 0 0 1.8 3h11.4a2 2 0 0 0 1.8-3L14 9.5V2" />
    <path d="M8.5 2h7" />
    <path d="M6.8 15h10.4" />
  </>,
);

export const GridIcon = makeIcon(
  <>
    <rect height="7" rx="1" width="7" x="3" y="3" />
    <rect height="7" rx="1" width="7" x="14" y="3" />
    <rect height="7" rx="1" width="7" x="14" y="14" />
    <rect height="7" rx="1" width="7" x="3" y="14" />
  </>,
);

export const BookIcon = makeIcon(
  <>
    <path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20" />
    <path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z" />
  </>,
);

export const FileIcon = makeIcon(
  <>
    <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
    <polyline points="14 2 14 8 20 8" />
  </>,
);

export const PeopleIcon = makeIcon(
  <>
    <path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2" />
    <circle cx="9" cy="7" r="4" />
    <path d="M23 21v-2a4 4 0 0 0-3-3.87" />
    <path d="M16 3.13a4 4 0 0 1 0 7.75" />
  </>,
);

export const TvIcon = makeIcon(
  <>
    <rect height="15" rx="2" width="20" x="2" y="7" />
    <polyline points="17 2 12 7 7 2" />
  </>,
);

export const RadioIcon = makeIcon(
  <>
    <circle cx="12" cy="12" r="2" />
    <path d="M16.24 7.76a6 6 0 0 1 0 8.49" />
    <path d="M7.76 16.24a6 6 0 0 1 0-8.49" />
    <path d="M19.07 4.93a10 10 0 0 1 0 14.14" />
    <path d="M4.93 19.07a10 10 0 0 1 0-14.14" />
  </>,
);

export const SlidersIcon = makeIcon(
  <>
    <path d="M21 4h-7" />
    <path d="M10 4H3" />
    <path d="M21 12h-9" />
    <path d="M8 12H3" />
    <path d="M21 20h-5" />
    <path d="M12 20H3" />
    <path d="M14 2v4" />
    <path d="M8 10v4" />
    <path d="M16 18v4" />
  </>,
);

export const HeartIcon = makeIcon(
  <path d="M20.84 4.61a5.5 5.5 0 0 0-7.78 0L12 5.67l-1.06-1.06a5.5 5.5 0 0 0-7.78 7.78l1.06 1.06L12 21.23l7.78-7.78 1.06-1.06a5.5 5.5 0 0 0 0-7.78z" />,
);

export const InfoIcon = makeIcon(
  <>
    <circle cx="12" cy="12" r="10" />
    <path d="M12 16v-4" />
    <path d="M12 8h.01" />
  </>,
);

export const ArrowUpIcon = makeIcon(
  <>
    <path d="M12 19V5" />
    <path d="m5 12 7-7 7 7" />
  </>,
);

export const ChevronLeftIcon = makeIcon(<polyline points="15 18 9 12 15 6" />);
export const ChevronRightIcon = makeIcon(<polyline points="9 18 15 12 9 6" />);
export const ChevronDownIcon = makeIcon(<polyline points="6 9 12 15 18 9" />);

export const ExternalLinkIcon = makeIcon(
  <>
    <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6" />
    <polyline points="15 3 21 3 21 9" />
    <path d="M10 14 21 3" />
  </>,
);

export const CopyIcon = makeIcon(
  <>
    <rect height="13" rx="2" width="13" x="9" y="9" />
    <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" />
  </>,
);

export const DownloadIcon = makeIcon(
  <>
    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
    <polyline points="7 10 12 15 17 10" />
    <path d="M12 15V3" />
  </>,
);

export const ClockIcon = makeIcon(
  <>
    <circle cx="12" cy="12" r="10" />
    <polyline points="12 6 12 12 16 14" />
  </>,
);

export const ShieldIcon = makeIcon(<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />);

export const LanguagesIcon = makeIcon(
  <>
    <path d="m5 8 6 6" />
    <path d="m4 14 6-6 2-3" />
    <path d="M2 5h12" />
    <path d="M7 2h1" />
    <path d="m22 22-5-10-5 10" />
    <path d="M14 18h6" />
  </>,
);

export const MagnetIcon = makeIcon(
  <>
    <path d="M6 3v8a6 6 0 0 0 12 0V3h-4v8a2 2 0 0 1-4 0V3H6z" />
    <path d="M6 7h4" />
    <path d="M14 7h4" />
  </>,
);

export const AlertIcon = makeIcon(
  <>
    <path d="M10.29 3.86 1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z" />
    <path d="M12 9v4" />
    <path d="M12 17h.01" />
  </>,
);

export const SparkIcon = makeIcon(
  <path d="M12 2c.8 5.2 3.8 8.2 10 10-6.2 1.8-9.2 4.8-10 10-.8-5.2-3.8-8.2-10-10 6.2-1.8 9.2-4.8 10-10z" />,
);

export const CalendarIcon = makeIcon(
  <>
    <rect height="18" rx="2" width="18" x="3" y="4" />
    <path d="M16 2v4" />
    <path d="M8 2v4" />
    <path d="M3 10h18" />
  </>,
);

export const FilmIcon = makeIcon(
  <>
    <rect height="20" rx="2.18" width="20" x="2" y="2" />
    <path d="M7 2v20" />
    <path d="M17 2v20" />
    <path d="M2 12h20" />
    <path d="M2 7h5" />
    <path d="M2 17h5" />
    <path d="M17 17h5" />
    <path d="M17 7h5" />
  </>,
);

export const PackageIcon = makeIcon(
  <>
    <path d="m16.5 9.4-9-5.19" />
    <path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z" />
    <polyline points="3.27 6.96 12 12.01 20.73 6.96" />
    <path d="M12 22.08V12" />
  </>,
);

export const CodeIcon = makeIcon(
  <>
    <polyline points="16 18 22 12 16 6" />
    <polyline points="8 6 2 12 8 18" />
  </>,
);

export const CheckIcon = makeIcon(<polyline points="20 6 9 17 4 12" />);

export const TagIcon = makeIcon(
  <>
    <path d="M20.59 13.41 12 22l-9-9V4a1 1 0 0 1 1-1h9l7.59 7.59a2 2 0 0 1 0 2.82z" />
    <circle cx="7.5" cy="7.5" r="1" />
  </>,
);

export const DotIcon = makeIcon(<circle cx="12" cy="12" fill="currentColor" r="4" stroke="none" />);

export const SpinnerIcon = makeIcon(
  <>
    <circle cx="12" cy="12" opacity="0.25" r="9" />
    <path d="M21 12a9 9 0 0 0-9-9" />
  </>,
);

export const CookieIcon = makeIcon(
  <>
    <path d="M21 11.5a8.38 8.38 0 0 1-.9 3.8 8.5 8.5 0 0 1-7.6 4.7 8.38 8.38 0 0 1-3.8-.9L3 21l1.9-5.7a8.38 8.38 0 0 1-.9-3.8 8.5 8.5 0 0 1 4.7-7.6 8.38 8.38 0 0 1 3.8-.9h.5a9 9 0 0 0 8 8v.5z" />
    <path d="M9 10h.01" />
    <path d="M14 8h.01" />
    <path d="M16 13h.01" />
    <path d="M11 15h.01" />
  </>,
);

export const KeyboardIcon = makeIcon(
  <>
    <rect height="12" rx="2" width="20" x="2" y="6" />
    <path d="M6 10h.01" />
    <path d="M10 10h.01" />
    <path d="M14 10h.01" />
    <path d="M18 10h.01" />
    <path d="M8 14h8" />
  </>,
);

export const KeyIcon = makeIcon(
  <>
    <circle cx="7.5" cy="15.5" r="4.5" />
    <path d="m10.85 12.15 8.9-8.9" />
    <path d="M18 5l2 2" />
    <path d="M15 8l2 2" />
  </>,
);

export const LinkIcon = makeIcon(
  <>
    <path d="M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71" />
    <path d="M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71" />
  </>,
);

export const SunIcon = makeIcon(
  <>
    <circle cx="12" cy="12" r="4" />
    <path d="M12 2v2" />
    <path d="M12 20v2" />
    <path d="m4.93 4.93 1.41 1.41" />
    <path d="m17.66 17.66 1.41 1.41" />
    <path d="M2 12h2" />
    <path d="M20 12h2" />
    <path d="m6.34 17.66-1.41 1.41" />
    <path d="m19.07 4.93-1.41 1.41" />
  </>,
);

export const MoonIcon = makeIcon(<path d="M12 3a6 6 0 0 0 9 9 9 9 0 1 1-9-9z" />);

export const StarIcon = makeIcon(
  <polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2" />,
);

export const SwapIcon = makeIcon(
  <>
    <path d="M8 3 4 7l4 4" />
    <path d="M4 7h16" />
    <path d="m16 21 4-4-4-4" />
    <path d="M20 17H4" />
  </>,
);

export const RefreshIcon = makeIcon(
  <>
    <path d="M3 12a9 9 0 0 1 9-9 9.75 9.75 0 0 1 6.74 2.74L21 8" />
    <path d="M21 3v5h-5" />
    <path d="M21 12a9 9 0 0 1-9 9 9.75 9.75 0 0 1-6.74-2.74L3 16" />
    <path d="M8 16H3v5" />
  </>,
);

export const LightbulbIcon = makeIcon(
  <>
    <path d="M9 18h6" />
    <path d="M10 22h4" />
    <path d="M12 2a7 7 0 0 0-4 12.7c.6.5 1 1.4 1 2.3h6c0-.9.4-1.8 1-2.3A7 7 0 0 0 12 2z" />
  </>,
);

export const MonitorIcon = makeIcon(
  <>
    <rect height="14" rx="2" width="20" x="2" y="3" />
    <path d="M8 21h8" />
    <path d="M12 17v4" />
    <path d="M12 13l-2.5-3.5L12 6l2.5 3.5L12 13z" fill="currentColor" stroke="none" />
  </>,
);

export const BarChartIcon = makeIcon(
  <>
    <path d="M3 3v18h18" />
    <rect height="6" rx="0.5" width="3" x="7" y="12" />
    <rect height="10" rx="0.5" width="3" x="12" y="8" />
    <rect height="13" rx="0.5" width="3" x="17" y="5" />
  </>,
);

export const CenterIcon = makeIcon(
  <>
    <rect height="18" rx="2" width="18" x="3" y="3" />
    <rect fill="currentColor" height="6" rx="1" stroke="none" width="8" x="8" y="9" />
  </>,
);

export const TerminalIcon = makeIcon(
  <>
    <polyline points="4 17 10 11 4 5" />
    <path d="m12 19 8 0" />
  </>,
);

// ------------------------------------------------------------------ brand

export function BrandMark({ className }: IconProps) {
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

// -------------------------------------------------------------- categories

const CATEGORY_ICONS: Record<string, (props: IconProps) => ReactNode> = {
  apps: GridIcon,
  dictionaries: BookIcon,
  files: FileIcon,
  general: SearchIcon,
  images: ImageIcon,
  it: LayersIcon,
  map: LocationIcon,
  music: MusicIcon,
  news: NewsIcon,
  radio: RadioIcon,
  science: FlaskIcon,
  "social media": PeopleIcon,
  TV: TvIcon,
  videos: PlayIcon,
};

export function CategoryIcon({ category, className }: IconProps & { category: string }) {
  const Icon = CATEGORY_ICONS[category] ?? GlobeIcon;
  return <Icon className={className} />;
}
