// SPDX-License-Identifier: Apache-2.0 WITH Commons-Clause-1.0

/**
 * Icon set: lucide-backed stroke icons on the 24x24 grid. Exported names
 * are the stable API — usage sites import `*Icon` from here and never see
 * lucide directly. Only BrandMark (the logo) is hand-drawn.
 */

import {
  AlertTriangle,
  AlignCenterVertical,
  ArrowDown,
  ArrowLeftRight,
  ArrowUp,
  Book,
  Calendar,
  ChartColumn,
  Check,
  ChevronDown,
  ChevronLeft,
  ChevronRight,
  Clock,
  Code,
  Cookie,
  Download,
  ExternalLink,
  FileText,
  Film,
  FlaskConical,
  Globe,
  GripVertical,
  Heart,
  Image,
  Info,
  Key,
  Keyboard,
  Languages,
  Layers,
  LayoutGrid,
  Lightbulb,
  Link,
  LoaderCircle,
  type LucideProps,
  Magnet,
  MapPin,
  Monitor,
  Moon,
  Music,
  Newspaper,
  Package,
  Pause,
  Play,
  Radio,
  RefreshCw,
  Search,
  Shield,
  SlidersHorizontal,
  Sparkle,
  Star,
  Sun,
  Tag,
  Terminal,
  Tv,
  Users,
  X,
} from "lucide-react";
import type { ComponentType, ReactNode } from "react";

interface IconProps {
  className?: string;
}

/** lucide with the theme defaults: decorative (no a11y noise), inherits
    text color, keeps the stroke style the whole UI is drawn with. */
function lucide(Icon: ComponentType<LucideProps>) {
  return function LucideIcon({ className }: IconProps) {
    return <Icon aria-hidden className={className} focusable="false" />;
  };
}

export const AlertIcon = lucide(AlertTriangle);
export const ArrowDownIcon = lucide(ArrowDown);
export const ArrowUpIcon = lucide(ArrowUp);
export const BarChartIcon = lucide(ChartColumn);
export const BookIcon = lucide(Book);
export const CalendarIcon = lucide(Calendar);
export const CenterIcon = lucide(AlignCenterVertical);
export const CheckIcon = lucide(Check);
export const ChevronDownIcon = lucide(ChevronDown);
export const ChevronLeftIcon = lucide(ChevronLeft);
export const ChevronRightIcon = lucide(ChevronRight);
export const ClockIcon = lucide(Clock);
export const CloseIcon = lucide(X);
export const CodeIcon = lucide(Code);
export const CookieIcon = lucide(Cookie);
export const DownloadIcon = lucide(Download);
export const ExternalLinkIcon = lucide(ExternalLink);
export const FileIcon = lucide(FileText);
export const FilmIcon = lucide(Film);
export const FlaskIcon = lucide(FlaskConical);
export const GlobeIcon = lucide(Globe);
export const GridIcon = lucide(LayoutGrid);
export const GripVerticalIcon = lucide(GripVertical);
export const HeartIcon = lucide(Heart);
export const ImageIcon = lucide(Image);
export const InfoIcon = lucide(Info);
export const KeyIcon = lucide(Key);
export const KeyboardIcon = lucide(Keyboard);
export const LanguagesIcon = lucide(Languages);
export const LayersIcon = lucide(Layers);
export const LightbulbIcon = lucide(Lightbulb);
export const LinkIcon = lucide(Link);
export const LocationIcon = lucide(MapPin);
export const MagnetIcon = lucide(Magnet);
export const MonitorIcon = lucide(Monitor);
export const MoonIcon = lucide(Moon);
export const MusicIcon = lucide(Music);
export const NewsIcon = lucide(Newspaper);
export const PackageIcon = lucide(Package);
export const PauseIcon = lucide(Pause);
export const PeopleIcon = lucide(Users);
export const PlayIcon = lucide(Play);
export const RadioIcon = lucide(Radio);
export const RefreshIcon = lucide(RefreshCw);
export const SearchIcon = lucide(Search);
export const ShieldIcon = lucide(Shield);
export const SlidersIcon = lucide(SlidersHorizontal);
export const SparkIcon = lucide(Sparkle);
export const SpinnerIcon = lucide(LoaderCircle);
export const StarIcon = lucide(Star);
export const SunIcon = lucide(Sun);
export const SwapIcon = lucide(ArrowLeftRight);
export const TagIcon = lucide(Tag);
export const TerminalIcon = lucide(Terminal);
export const TvIcon = lucide(Tv);

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
