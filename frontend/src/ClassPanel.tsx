import { Plus } from "lucide-react";
import type { Category } from "./types";
import IconButton from "./IconButton";
export default function ClassPanel(p: {
  categories: Category[];
  categoryId: number;
  onSelect: (id: number) => void;
  onEdit: (id: number) => void;
  onCreate: () => void;
  createTitle?: string;
}) {
  return (
    <section className="classes-panel">
      <div className="panel-label">
        <span>Classes</span>
        <IconButton
          icon={Plus}
          title={p.createTitle || "Create class"}
          onClick={p.onCreate}
        />
      </div>
      <div className="category-chips">
        {p.categories.map((c) => (
          <button
            key={c.id}
            className={p.categoryId === c.id ? "active" : ""}
            onClick={() => p.onSelect(c.id)}
            onDoubleClick={() => p.onEdit(c.id)}
            title={`${c.name} · double-click to edit`}
          >
            <i style={{ background: c.color }} />
            {c.name}
          </button>
        ))}
        {!p.categories.length && (
          <button onClick={p.onCreate}>
            <Plus size={12} />
            Create first class
          </button>
        )}
      </div>
    </section>
  );
}
