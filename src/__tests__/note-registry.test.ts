import { NoteRegistry } from "../note-registry";
import { NoteMeta } from "../types";

describe("NoteRegistry", () => {
  describe("ensureId", () => {
    it("mints a new id for an unknown path", () => {
      const reg = new NoteRegistry();
      const id = reg.ensureId("a.md", 100);
      expect(id).toBeTruthy();
      expect(reg.getId("a.md")).toBe(id);
      expect(reg.getPath(id)).toBe("a.md");
      expect(reg.getMtime(id)).toBe(100);
    });

    it("returns the existing id on repeated calls", () => {
      const reg = new NoteRegistry();
      const first = reg.ensureId("a.md", 100);
      const second = reg.ensureId("a.md", 200);
      expect(second).toBe(first);
      expect(reg.getMtime(first)).toBe(200);
    });
  });

  describe("rename", () => {
    it("keeps the same id when a path is renamed", () => {
      const reg = new NoteRegistry();
      const id = reg.ensureId("old.md", 100);
      reg.rename("old.md", "new.md");
      expect(reg.getId("old.md")).toBeNull();
      expect(reg.getId("new.md")).toBe(id);
      expect(reg.getPath(id)).toBe("new.md");
    });

    it("is a no-op when the old path is unknown", () => {
      const reg = new NoteRegistry();
      reg.rename("missing.md", "new.md");
      expect(reg.getId("new.md")).toBeNull();
    });
  });

  describe("drop", () => {
    it("removes a path and returns its id", () => {
      const reg = new NoteRegistry();
      const id = reg.ensureId("a.md", 100);
      const dropped = reg.drop("a.md");
      expect(dropped).toBe(id);
      expect(reg.getId("a.md")).toBeNull();
      expect(reg.getPath(id)).toBeNull();
    });

    it("returns null for an unknown path", () => {
      const reg = new NoteRegistry();
      expect(reg.drop("missing.md")).toBeNull();
    });
  });

  describe("setMtime", () => {
    it("updates the mtime for an existing id", () => {
      const reg = new NoteRegistry();
      const id = reg.ensureId("a.md", 100);
      reg.setMtime(id, 500);
      expect(reg.getMtime(id)).toBe(500);
    });
  });

  describe("toNoteMap / fromNoteMap", () => {
    it("round-trips through the persisted shape", () => {
      const reg = new NoteRegistry();
      const idA = reg.ensureId("a.md", 100);
      const idB = reg.ensureId("b.md", 200);

      const serialized = reg.toNoteMap();
      expect(serialized[idA]).toEqual({ path: "a.md", mtime: 100 });
      expect(serialized[idB]).toEqual({ path: "b.md", mtime: 200 });

      const restored = new NoteRegistry(serialized);
      expect(restored.getId("a.md")).toBe(idA);
      expect(restored.getId("b.md")).toBe(idB);
      expect(restored.getMtime(idA)).toBe(100);
    });
  });

  describe("reconcile", () => {
    it("mints ids for new files in the vault", () => {
      const reg = new NoteRegistry();
      const result = reg.reconcile([{ path: "new.md", mtime: 10, size: 5 }]);
      expect(reg.getId("new.md")).toBeTruthy();
      expect(result.added).toEqual([reg.getId("new.md")]);
      expect(result.reattached).toEqual([]);
      expect(result.dropped).toEqual([]);
    });

    it("preserves ids for files that still exist", () => {
      const initial: Record<string, NoteMeta> = {
        "id-1": { path: "a.md", mtime: 100 },
      };
      const reg = new NoteRegistry(initial);
      const result = reg.reconcile([{ path: "a.md", mtime: 100, size: 5 }]);
      expect(reg.getId("a.md")).toBe("id-1");
      expect(result.added).toEqual([]);
      expect(result.dropped).toEqual([]);
    });

    it("drops ids whose path no longer exists when no match found", () => {
      const initial: Record<string, NoteMeta> = {
        "id-1": { path: "gone.md", mtime: 100 },
      };
      const reg = new NoteRegistry(initial);
      const result = reg.reconcile([]);
      expect(reg.getPath("id-1")).toBeNull();
      expect(result.dropped).toEqual(["id-1"]);
    });

    it("reattaches an orphaned id to a new path with matching size", () => {
      const initial: Record<string, NoteMeta> = {
        "id-1": { path: "old-name.md", mtime: 100, size: 42 } as NoteMeta & { size?: number },
      };
      // noteMeta type is {path, mtime} — we pass size as extra metadata for reconciliation.
      const reg = new NoteRegistry(initial, {
        lastSeenSize: { "id-1": 42 },
      });
      const result = reg.reconcile([{ path: "new-name.md", mtime: 100, size: 42 }]);
      expect(reg.getId("new-name.md")).toBe("id-1");
      expect(reg.getPath("id-1")).toBe("new-name.md");
      expect(result.reattached).toEqual([{ id: "id-1", newPath: "new-name.md" }]);
      expect(result.dropped).toEqual([]);
    });

    it("drops orphans when reattach is ambiguous (two candidates match)", () => {
      const reg = new NoteRegistry(
        { "id-1": { path: "old.md", mtime: 100 } },
        { lastSeenSize: { "id-1": 42 } },
      );
      const result = reg.reconcile([
        { path: "new-a.md", mtime: 100, size: 42 },
        { path: "new-b.md", mtime: 100, size: 42 },
      ]);
      expect(result.reattached).toEqual([]);
      expect(result.dropped).toContain("id-1");
      // both new paths should have been given fresh ids.
      expect(reg.getId("new-a.md")).toBeTruthy();
      expect(reg.getId("new-b.md")).toBeTruthy();
      expect(reg.getId("new-a.md")).not.toBe("id-1");
    });
  });
});
