// SPDX-License-Identifier: Apache-2.0 WITH Commons-Clause-1.0

/**
 * Icon set: lucide-backed stroke icons on the 24x24 grid, plus a handful of
 * custom marks (brand, spinner, monitor-pointer, center-frame, bar-chart)
 * that lucide does not express. Exported names are the stable API — usage
 * sites keep importing *Icon from here and never see lucide directly.
 */

import {
  AlertTriangle as LucideAlert,
  ArrowDown as LucideArrowDown,
  ArrowUp as LucideArrowUp,
  Book as LucideBook,
  Calendar as LucideCalendar,
  Check as LucideCheck,
  ChevronDown as LucideChevronDown,
  ChevronLeft as LucideChevronLeft,
  ChevronRight as LucideChevronRight,
  Clock as LucideClock,
  X as LucideClose,
  Code as LucideCode,
  Cookie as LucideCookie,
  Download as LucideDownload,
  ExternalLink as LucideExternalLink,
  FileText as LucideFile,
  Film as LucideFilm,
  FlaskConical as LucideFlask,
  Globe as LucideGlobe,
  LayoutGrid as LucideGrid,
  GripVertical as LucideGripVertical,
  Heart as LucideHeart,
  Image as LucideImage,
  Info as LucideInfo,
  Key as LucideKey,
  Keyboard as LucideKeyboard,
  Languages as LucideLanguages,
  Layers as LucideLayers,
  Lightbulb as LucideLightbulb,
  Link as LucideLink,
  MapPin as LucideLocation,
  Magnet as LucideMagnet,
  Monitor as LucideMonitorBase,
  Moon as LucideMoon,
  Music as LucideMusic,
  Newspaper as LucideNews,
  Package as LucidePackage,
  Pause as LucidePause,
  Users as LucidePeople,
  Play as LucidePlay,
  type LucideProps,
  Radio as LucideRadio,
  RefreshCw as LucideRefresh,
  Search as LucideSearch,
  Shield as LucideShield,
  SlidersHorizontal as LucideSliders,
  Sparkle as LucideSpark,
  Star as LucideStar,
  Sun as LucideSun,
  ArrowLeftRight as LucideSwap,
  Tag as LucideTag,
  Terminal as LucideTerminal,
  Tv as LucideTv,
} from "lucide-react";
import type { ComponentType, ReactNode } from "react";

interface IconProps {
  className?: string;
}

/** lucide with the theme defaults: decorative (no a11y noise), inherits text
    color, keeps the stroke style the whole UI is drawn with. */
function lucide(Icon: ComponentType<LucideProps>) {
  return function LucideIcon({ className }: IconProps) {
    return <Icon aria-hidden className={className} focusable="false" />;
  };
}

// ------------------------------------------------------------- lucide icons

export const SearchIcon = lucide(LucideSearch);
export const CloseIcon = lucide(LucideClose);
export const GlobeIcon = lucide(LucideGlobe);
export const ImageIcon = lucide(LucideImage);
export const PlayIcon = lucide(LucidePlay);
export const PauseIcon = lucide(LucidePause);
export const NewsIcon = lucide(LucideNews);
export const LayersIcon = lucide(LucideLayers);
export const LocationIcon = lucide(LucideLocation);
export const MusicIcon = lucide(LucideMusic);
export const FlaskIcon = lucide(LucideFlask);
export const GridIcon = lucide(LucideGrid);
export const BookIcon = lucide(LucideBook);
export const FileIcon = lucide(LucideFile);
export const PeopleIcon = lucide(LucidePeople);
export const TvIcon = lucide(LucideTv);
export const RadioIcon = lucide(LucideRadio);
export const SlidersIcon = lucide(LucideSliders);
export const HeartIcon = lucide(LucideHeart);
export const InfoIcon = lucide(LucideInfo);
export const ArrowUpIcon = lucide(LucideArrowUp);
export const ArrowDownIcon = lucide(LucideArrowDown);
export const ChevronLeftIcon = lucide(LucideChevronLeft);
export const ChevronRightIcon = lucide(LucideChevronRight);
export const ChevronDownIcon = lucide(LucideChevronDown);
export const ExternalLinkIcon = lucide(LucideExternalLink);
export const DownloadIcon = lucide(LucideDownload);
export const ClockIcon = lucide(LucideClock);
export const ShieldIcon = lucide(LucideShield);
export const LanguagesIcon = lucide(LucideLanguages);
export const MagnetIcon = lucide(LucideMagnet);
export const AlertIcon = lucide(LucideAlert);
export const SparkIcon = lucide(LucideSpark);
export const CalendarIcon = lucide(LucideCalendar);
export const FilmIcon = lucide(LucideFilm);
export const PackageIcon = lucide(LucidePackage);
export const CodeIcon = lucide(LucideCode);
export const CheckIcon = lucide(LucideCheck);
export const TagIcon = lucide(LucideTag);
export const GripVerticalIcon = lucide(LucideGripVertical);
export const CookieIcon = lucide(LucideCookie);
export const KeyboardIcon = lucide(LucideKeyboard);
export const KeyIcon = lucide(LucideKey);
export const LinkIcon = lucide(LucideLink);
export const SunIcon = lucide(LucideSun);
export const MoonIcon = lucide(LucideMoon);
export const StarIcon = lucide(LucideStar);
export const SwapIcon = lucide(LucideSwap);
export const RefreshIcon = lucide(LucideRefresh);
export const LightbulbIcon = lucide(LucideLightbulb);
export const TerminalIcon = lucide(LucideTerminal);

// ------------------------------------------------- custom marks (no lucide)

/** Loading spinner: track + arc — lucide's LoaderCircle has no track. */
export const SpinnerIcon = function SpinnerIcon({ className }: IconProps) {
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
      viewBox="0 0 24 24"
    >
      <circle cx="12" cy="12" opacity="0.25" r="9" />
      <path d="M21 12a9 9 0 0 0-9-9" />
    </svg>
  );
};

/** Theme monitor: lucide's Monitor lacks the filled "active display" pointer. */
export const MonitorIcon = function MonitorIcon({ className }: IconProps) {
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
      viewBox="0 0 24 24"
    >
      <rect height="14" rx="2" width="20" x="2" y="3" />
      <path d="M8 21h8" />
      <path d="M12 17v4" />
      <path d="M12 13l-2.5-3.5L12 6l2.5 3.5L12 13z" fill="currentColor" stroke="none" />
    </svg>
  );
};

/** Center alignment: frame with a filled centered rect. */
export const CenterIcon = function CenterIcon({ className }: IconProps) {
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
      viewBox="0 0 24 24"
    >
      <rect height="18" rx="2" width="18" x="3" y="3" />
      <rect fill="currentColor" height="6" rx="1" stroke="none" width="8" x="8" y="9" />
    </svg>
  );
};

/** Stats chart: axis + column bars. */
export const BarChartIcon = function BarChartIcon({ className }: IconProps) {
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
      viewBox="0 0 24 24"
    >
      <path d="M3 3v18h18" />
      <rect height="6" rx="0.5" width="3" x="7" y="12" />
      <rect height="10" rx="0.5" width="3" x="12" y="8" />
      <rect height="13" rx="0.5" width="3" x="17" y="5" />
    </svg>
  );
};

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
  packages: PackageIcon,
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
