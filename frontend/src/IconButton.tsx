import { useId } from "react";
import type { LucideIcon } from "lucide-react";
export default function IconButton({
  icon: Icon,
  title,
  onClick,
  active = false,
  disabled = false,
}: {
  icon: LucideIcon;
  title: string;
  onClick: () => void;
  active?: boolean;
  disabled?: boolean;
}) {
  const tooltipId = useId();
  return (
    <span className="tool-control">
      <button
        type="button"
        className={`icon-button ${active ? "active" : ""}`}
        aria-label={title}
        aria-describedby={tooltipId}
        aria-pressed={active}
        onClick={onClick}
        disabled={disabled}
      >
        <Icon size={18} strokeWidth={1.7} />
      </button>
      <span className="tool-tooltip" role="tooltip" id={tooltipId}>
        {title}
      </span>
    </span>
  );
}
