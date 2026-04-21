import { EdgeStore } from "../edge-store";
import { JobQueue } from "../job-queue";
import { createEmptyPersistedState, PersistedState } from "../types";

function makeStore(threshold = 0.7) {
  const state: PersistedState = { ...createEmptyPersistedState(), similarityThreshold: threshold };
  const onPersist = jest.fn();
  const store = new EdgeStore(state, onPersist);
  // Add a source file so allSourceIds returns something.
  store.setEdgesForFile("id-a", [], 100);
  store.setEdgesForFile("id-b", [], 200);
  return { store, onPersist };
}

function makeQueue() {
  const onProcess = jest.fn();
  return { queue: new JobQueue(onProcess), onProcess };
}

describe("threshold slider behaviour", () => {
  beforeEach(() => jest.useFakeTimers());
  afterEach(() => jest.useRealTimers());

  describe("raising threshold (setThresholdAndPrune)", () => {
    it("prunes edges below the new value", () => {
      const { store } = makeStore(0.5);
      const phrase = { phrase: "x", score: 0, startOffset: 0, endOffset: 1, spanId: "0:1" };
      store.setEdgesForFile("id-a", [
        { sourcePath: "a.md", sourceId: "id-a", phrase, targetPath: "T", targetId: "id-t", similarity: 0.9 },
        { sourcePath: "a.md", sourceId: "id-a", phrase, targetPath: "U", targetId: "id-u", similarity: 0.4 },
      ], 100);
      store.setThresholdAndPrune(0.8);
      // Lower to 0 — pruned edges should be gone from storage.
      store.setThresholdNoPrune(0);
      expect(store.getEdgesForFile("id-a")).toHaveLength(1);
      expect(store.getEdgesForFile("id-a")[0].similarity).toBe(0.9);
    });

    it("does not trigger a queue reprocess", () => {
      const { store } = makeStore();
      const { queue, onProcess } = makeQueue();
      store.setThresholdAndPrune(0.9);
      jest.advanceTimersByTime(1000);
      expect(onProcess).not.toHaveBeenCalled();
    });
  });

  describe("lowering threshold (setThresholdNoPrune + forceEnqueue)", () => {
    it("does not prune existing edges", () => {
      const { store } = makeStore(0.9);
      const phrase = { phrase: "x", score: 0, startOffset: 0, endOffset: 1, spanId: "0:1" };
      store.setEdgesForFile("id-a", [
        { sourcePath: "a.md", sourceId: "id-a", phrase, targetPath: "T", targetId: "id-t", similarity: 0.95 },
      ], 100);
      store.setThresholdNoPrune(0.5);
      // Edge was stored above old threshold, still present.
      expect(store.getEdgesForFile("id-a")).toHaveLength(1);
    });

    it("caller triggers forceEnqueue for each source file", () => {
      const { store } = makeStore();
      const { queue, onProcess } = makeQueue();

      // First process both files so cooldown would block normal enqueue.
      queue.enqueue("a.md", "process");
      queue.enqueue("b.md", "process");
      jest.advanceTimersByTime(600);
      expect(onProcess).toHaveBeenCalledTimes(2);

      // Simulate lowering threshold — caller must use forceEnqueue.
      store.setThresholdNoPrune(0.3);
      for (const id of store.allSourceIds()) {
        queue.forceEnqueue(id, "process"); // id doubles as path in this test
      }
      jest.advanceTimersByTime(600);
      // Should have processed again despite cooldown.
      expect(onProcess).toHaveBeenCalledTimes(4);
    });
  });

  describe("ProgressToast counter logic", () => {
    it("decrements on each setEdgesForFile and reaches zero", () => {
      const { store } = makeStore();
      const sourceIds = store.allSourceIds(); // ["id-a", "id-b"]
      let remaining = sourceIds.length;
      const onDone = jest.fn();

      store.subscribe(() => {
        remaining--;
        if (remaining <= 0) onDone();
      });

      store.setEdgesForFile("id-a", [], 101);
      store.setEdgesForFile("id-b", [], 201);

      expect(onDone).toHaveBeenCalledTimes(1);
    });
  });
});
