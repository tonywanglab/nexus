import { Vault, TFile } from "obsidian";
import { NoteRegistry } from "../note-registry";
import { EdgeStore } from "../edge-store";
import { JobQueue } from "../job-queue";
import { createEmptyPersistedState, PersistedState } from "../types";
import { initializeFromCache } from "../initialize";

function makeFile(path: string, mtime: number, size = 10): TFile {
  return new TFile(path, { mtime, size });
}

function setup(persisted?: Partial<PersistedState>) {
  const vault = new Vault() as any;
  const state: PersistedState = { ...createEmptyPersistedState(), ...persisted };
  const registry = new NoteRegistry(state.notes);
  const store = new EdgeStore(state, jest.fn());
  const onProcess = jest.fn();
  const queue = new JobQueue(onProcess);
  return { vault, registry, store, queue, onProcess };
}

describe("initializeFromCache", () => {
  beforeEach(() => jest.useFakeTimers());
  afterEach(() => jest.useRealTimers());

  it("enqueues a file whose mtime has changed", async () => {
    const { vault, registry, store, queue, onProcess } = setup({
      notes: { "id-a": { path: "a.md", mtime: 100 } },
    });
    vault._addFile(makeFile("a.md", 200)); // newer mtime
    await initializeFromCache(vault, registry, store, queue);
    jest.advanceTimersByTime(600);
    expect(onProcess).toHaveBeenCalledTimes(1);
    expect(onProcess.mock.calls[0][0].filePath).toBe("a.md");
  });

  it("does not enqueue a file whose mtime matches the cache", async () => {
    const { vault, registry, store, queue, onProcess } = setup({
      notes: { "id-a": { path: "a.md", mtime: 100 } },
    });
    vault._addFile(makeFile("a.md", 100)); // same mtime
    // Also store some cached edges so getEdgesForFile would return them
    store.setEdgesForFile("id-a", [], 100);
    await initializeFromCache(vault, registry, store, queue);
    jest.advanceTimersByTime(600);
    expect(onProcess).not.toHaveBeenCalled();
  });

  it("enqueues a new file not in the cache", async () => {
    const { vault, registry, store, queue, onProcess } = setup();
    vault._addFile(makeFile("new.md", 50));
    await initializeFromCache(vault, registry, store, queue);
    jest.advanceTimersByTime(600);
    expect(onProcess).toHaveBeenCalledTimes(1);
    expect(onProcess.mock.calls[0][0].filePath).toBe("new.md");
  });

  it("drops edges for a file no longer in the vault", async () => {
    const phrase = { phrase: "hi", score: 0, startOffset: 0, endOffset: 2, spanId: "0:2" };
    const { vault, registry, store, queue } = setup({
      notes: { "id-gone": { path: "gone.md", mtime: 100 } },
      edges: {
        "id-gone": [
          {
            sourcePath: "gone.md", sourceId: "id-gone",
            phrase, targetPath: "X", targetId: "id-x",
            similarity: 0.9,
          },
        ],
      },
    });
    // vault has no files — "gone.md" is absent
    await initializeFromCache(vault, registry, store, queue);
    expect(store.getMtime("id-gone")).toBeNull();
  });

  it("mints a new id for a file absent from the registry", async () => {
    const { vault, registry, store, queue } = setup();
    vault._addFile(makeFile("fresh.md", 77));
    await initializeFromCache(vault, registry, store, queue);
    expect(registry.getId("fresh.md")).toBeTruthy();
  });
});
