import { randomUUID } from "crypto";
import { NoteId, NoteMeta } from "./types";

export interface VaultFileSnapshot {
  path: string;
  mtime: number;
  size: number;
}

export interface ReconcileResult {
  added: NoteId[];
  reattached: Array<{ id: NoteId; newPath: string }>;
  dropped: NoteId[];
}

export interface NoteRegistryExtras {
  //  Optional per-id byte size carried over from the previous session (used for rename reattach).
  lastSeenSize?: Record<NoteId, number>;
}

// maintains the bijective map between note file paths and stable ids.
// ids survive renames; paths are the mutable side.
//
// no event emission — callers orchestrate store updates after calling registry mutators.
export class NoteRegistry {
  private idToMeta = new Map<NoteId, NoteMeta>();
  private pathToId = new Map<string, NoteId>();
  private lastSeenSize = new Map<NoteId, number>();

  constructor(initial?: Record<NoteId, NoteMeta>, extras?: NoteRegistryExtras) {
    if (initial) {
      for (const [id, meta] of Object.entries(initial)) {
        this.idToMeta.set(id, { path: meta.path, mtime: meta.mtime });
        this.pathToId.set(meta.path, id);
      }
    }
    if (extras?.lastSeenSize) {
      for (const [id, size] of Object.entries(extras.lastSeenSize)) {
        this.lastSeenSize.set(id, size);
      }
    }
  }

  ensureId(path: string, mtime: number, size?: number): NoteId {
    const existing = this.pathToId.get(path);
    if (existing) {
      const meta = this.idToMeta.get(existing)!;
      meta.mtime = mtime;
      if (size !== undefined) this.lastSeenSize.set(existing, size);
      return existing;
    }
    const id = randomUUID();
    this.idToMeta.set(id, { path, mtime });
    this.pathToId.set(path, id);
    if (size !== undefined) this.lastSeenSize.set(id, size);
    return id;
  }

  rename(oldPath: string, newPath: string): void {
    const id = this.pathToId.get(oldPath);
    if (!id) return;
    const meta = this.idToMeta.get(id)!;
    meta.path = newPath;
    this.pathToId.delete(oldPath);
    this.pathToId.set(newPath, id);
  }

  drop(path: string): NoteId | null {
    const id = this.pathToId.get(path);
    if (!id) return null;
    this.pathToId.delete(path);
    this.idToMeta.delete(id);
    this.lastSeenSize.delete(id);
    return id;
  }

  getId(path: string): NoteId | null {
    return this.pathToId.get(path) ?? null;
  }

  getPath(id: NoteId): string | null {
    return this.idToMeta.get(id)?.path ?? null;
  }

  getMtime(id: NoteId): number | null {
    return this.idToMeta.get(id)?.mtime ?? null;
  }

  setMtime(id: NoteId, mtime: number): void {
    const meta = this.idToMeta.get(id);
    if (meta) meta.mtime = mtime;
  }

  setSize(id: NoteId, size: number): void {
    this.lastSeenSize.set(id, size);
  }

  getSize(id: NoteId): number | null {
    return this.lastSeenSize.get(id) ?? null;
  }

  allIds(): NoteId[] {
    return [...this.idToMeta.keys()];
  }

  toNoteMap(): Record<NoteId, NoteMeta> {
    const out: Record<NoteId, NoteMeta> = {};
    for (const [id, meta] of this.idToMeta) {
      out[id] = { path: meta.path, mtime: meta.mtime };
    }
    return out;
  }

  toSizeMap(): Record<NoteId, number> {
    const out: Record<NoteId, number> = {};
    for (const [id, size] of this.lastSeenSize) out[id] = size;
    return out;
  }

  // diffs the vault against persisted state. Mints ids for new paths, attempts
  // best-effort reattach of orphaned ids by size-match (exactly one candidate),
  // drops the rest.
  reconcile(files: VaultFileSnapshot[]): ReconcileResult {
    const result: ReconcileResult = { added: [], reattached: [], dropped: [] };

    const currentPaths = new Set(files.map((f) => f.path));
    const orphanIds: NoteId[] = [];
    for (const id of this.idToMeta.keys()) {
      if (!currentPaths.has(this.idToMeta.get(id)!.path)) orphanIds.push(id);
    }

    const unclaimedFiles = files.filter((f) => !this.pathToId.has(f.path));

    // best-effort reattach: for each orphan, find unclaimed files matching its last-seen size.
    const claimedByReattach = new Set<string>();
    for (const id of orphanIds) {
      const size = this.lastSeenSize.get(id);
      if (size === undefined) continue;
      const candidates = unclaimedFiles.filter(
        (f) => f.size === size && !claimedByReattach.has(f.path),
      );
      if (candidates.length === 1) {
        const target = candidates[0];
        const meta = this.idToMeta.get(id)!;
        this.pathToId.delete(meta.path);
        meta.path = target.path;
        meta.mtime = target.mtime;
        this.pathToId.set(target.path, id);
        this.lastSeenSize.set(id, target.size);
        claimedByReattach.add(target.path);
        result.reattached.push({ id, newPath: target.path });
      }
    }

    // drop orphans that weren't reattached.
    for (const id of orphanIds) {
      if (result.reattached.some((r) => r.id === id)) continue;
      const meta = this.idToMeta.get(id)!;
      this.pathToId.delete(meta.path);
      this.idToMeta.delete(id);
      this.lastSeenSize.delete(id);
      result.dropped.push(id);
    }

    // mint ids for still-unclaimed files.
    for (const f of unclaimedFiles) {
      if (claimedByReattach.has(f.path)) continue;
      const id = this.ensureId(f.path, f.mtime, f.size);
      result.added.push(id);
    }

    return result;
  }
}
