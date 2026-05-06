import { EdgeStore } from "../edge-store";
import { PersistedState, CandidateEdge, createEmptyPersistedState } from "../types";

function makeEdge(
  phraseText: string,
  targetPath: string,
  similarity: number,
  sourceId = "id-src",
  targetId = "id-tgt",
): CandidateEdge {
  return {
    sourcePath: "src.md",
    sourceId,
    phrase: { phrase: phraseText, score: 0.1, startOffset: 0, endOffset: phraseText.length, spanId: `0:${phraseText.length}` },
    targetPath,
    targetId,
    similarity,
    matchType: "deterministic",
  };
}

function makeStore(initial?: Partial<PersistedState>, onPersist?: jest.Mock) {
  const state: PersistedState = { ...createEmptyPersistedState(), ...initial };
  const persist = onPersist ?? jest.fn();
  return { store: new EdgeStore(state, persist), persist };
}

describe("EdgeStore", () => {
  beforeEach(() => jest.useFakeTimers());
  afterEach(() => jest.useRealTimers());

  describe("setEdgesForFile / getEdgesForFile", () => {
    it("stores and retrieves edges sorted by similarity desc", () => {
      const { store } = makeStore({ similarityThreshold: 0 });
      const edges = [
        makeEdge("foo", "Foo", 0.8, "id-src", "id-1"),
        makeEdge("bar", "Bar", 0.95, "id-src", "id-2"),
        makeEdge("baz", "Baz", 0.6, "id-src", "id-3"),
      ];
      store.setEdgesForFile("id-src", edges, 100);
      const out = store.getEdgesForFile("id-src");
      expect(out.map((e) => e.similarity)).toEqual([0.95, 0.8, 0.6]);
    });

    it("returns empty array for unknown id", () => {
      const { store } = makeStore();
      expect(store.getEdgesForFile("unknown")).toEqual([]);
    });

    it("replaces previous edges on update", () => {
      const { store } = makeStore();
      store.setEdgesForFile("id-src", [makeEdge("foo", "Foo", 0.9)], 100);
      store.setEdgesForFile("id-src", [makeEdge("bar", "Bar", 0.8)], 200);
      const out = store.getEdgesForFile("id-src");
      expect(out).toHaveLength(1);
      expect(out[0].phrase.phrase).toBe("bar");
    });

    it("filters denied edges on setEdgesForFile", () => {
      const { store } = makeStore();
      store.denyEdge("id-src", "foo", "id-tgt");
      store.setEdgesForFile("id-src", [makeEdge("foo", "Foo", 0.9)], 100);
      expect(store.getEdgesForFile("id-src")).toHaveLength(0);
    });

    it("filters approved edges on setEdgesForFile", () => {
      const { store } = makeStore();
      store.approveEdge("id-src", "foo", "id-tgt");
      store.setEdgesForFile("id-src", [makeEdge("foo", "Foo", 0.9)], 100);
      expect(store.getEdgesForFile("id-src")).toHaveLength(0);
    });

    it("denial key is normalized (case/whitespace insensitive)", () => {
      const { store } = makeStore();
      store.denyEdge("id-src", "Neural Nets", "id-tgt");
      store.setEdgesForFile("id-src", [makeEdge("neural  nets", "Foo", 0.9)], 100);
      expect(store.getEdgesForFile("id-src")).toHaveLength(0);
    });
  });

  describe("threshold", () => {
    it("getEdgesForFile excludes edges below threshold without mutating storage", () => {
      const { store } = makeStore({ similarityThreshold: 0.8 });
      store.setEdgesForFile("id-src", [
        makeEdge("a", "A", 0.9),
        makeEdge("b", "B", 0.75),
      ], 100);
      expect(store.getEdgesForFile("id-src")).toHaveLength(1);
      expect(store.getEdgesForFile("id-src")[0].similarity).toBe(0.9);
    });

    it("setThresholdAndPrune removes stored edges below value", () => {
      const { store } = makeStore();
      store.setEdgesForFile("id-src", [
        makeEdge("a", "A", 0.9),
        makeEdge("b", "B", 0.5),
      ], 100);
      store.setThresholdAndPrune(0.8);
      // lower threshold — edges below 0.8 already deleted from storage
      store.setThresholdNoPrune(0.3);
      const out = store.getEdgesForFile("id-src");
      expect(out).toHaveLength(1);
      expect(out[0].similarity).toBe(0.9);
    });

    it("setThresholdNoPrune does not touch stored edges", () => {
      const { store } = makeStore();
      store.setEdgesForFile("id-src", [
        makeEdge("a", "A", 0.9),
        makeEdge("b", "B", 0.5),
      ], 100);
      store.setThresholdNoPrune(0.8);
      // raw storage still has both edges; getEdgesForFile applies threshold filter
      expect(store.getEdgesForFile("id-src")).toHaveLength(1);
      store.setThresholdNoPrune(0.3);
      expect(store.getEdgesForFile("id-src")).toHaveLength(2);
    });

    it("getThreshold returns current value", () => {
      const { store } = makeStore({ similarityThreshold: 0.7 });
      expect(store.getThreshold()).toBe(0.7);
      store.setThresholdAndPrune(0.9);
      expect(store.getThreshold()).toBe(0.9);
    });
  });

  describe("denyEdge", () => {
    it("removes the matching edge from the store", () => {
      const { store } = makeStore();
      store.setEdgesForFile("id-src", [makeEdge("foo", "Foo", 0.9)], 100);
      store.denyEdge("id-src", "foo", "id-tgt");
      expect(store.getEdgesForFile("id-src")).toHaveLength(0);
    });

    it("isDenied returns true after denial", () => {
      const { store } = makeStore();
      store.denyEdge("id-src", "foo", "id-tgt");
      expect(store.isDenied("id-src", "foo", "id-tgt")).toBe(true);
    });

    it("denial is file-scoped: different sourceId is not affected", () => {
      const { store } = makeStore();
      store.denyEdge("id-src", "foo", "id-tgt");
      expect(store.isDenied("id-other", "foo", "id-tgt")).toBe(false);
    });
  });

  describe("approveEdge", () => {
    it("removes the matching edge from the store", () => {
      const { store } = makeStore();
      store.setEdgesForFile("id-src", [makeEdge("foo", "Foo", 0.9)], 100);
      store.approveEdge("id-src", "foo", "id-tgt");
      expect(store.getEdgesForFile("id-src")).toHaveLength(0);
    });

    it("isApproved returns true after approval", () => {
      const { store } = makeStore();
      store.approveEdge("id-src", "foo", "id-tgt");
      expect(store.isApproved("id-src", "foo", "id-tgt")).toBe(true);
    });
  });

  describe("removeEdge", () => {
    it("removes only the specified edge", () => {
      const { store } = makeStore();
      store.setEdgesForFile("id-src", [
        makeEdge("foo", "Foo", 0.9, "id-src", "id-foo"),
        makeEdge("bar", "Bar", 0.8, "id-src", "id-bar"),
      ], 100);
      store.removeEdge("id-src", "foo", "id-foo");
      const out = store.getEdgesForFile("id-src");
      expect(out).toHaveLength(1);
      expect(out[0].phrase.phrase).toBe("bar");
    });
  });

  describe("dropNote", () => {
    it("removes all edges sourced from the dropped id", () => {
      const { store } = makeStore();
      store.setEdgesForFile("id-src", [makeEdge("foo", "Foo", 0.9)], 100);
      store.dropNote("id-src");
      expect(store.getEdgesForFile("id-src")).toHaveLength(0);
    });

    it("removes edges targeting the dropped id across all sources", () => {
      const { store } = makeStore();
      store.setEdgesForFile("id-src", [makeEdge("foo", "Foo", 0.9, "id-src", "id-gone")], 100);
      store.dropNote("id-gone");
      expect(store.getEdgesForFile("id-src")).toHaveLength(0);
    });
  });

  describe("subscribe", () => {
    it("notifies listeners on mutation", () => {
      const { store } = makeStore();
      const listener = jest.fn();
      store.subscribe(listener);
      store.setEdgesForFile("id-src", [makeEdge("foo", "Foo", 0.9)], 100);
      expect(listener).toHaveBeenCalledTimes(1);
    });

    it("unsubscribe stops notifications", () => {
      const { store } = makeStore();
      const listener = jest.fn();
      const unsub = store.subscribe(listener);
      unsub();
      store.setEdgesForFile("id-src", [makeEdge("foo", "Foo", 0.9)], 100);
      expect(listener).not.toHaveBeenCalled();
    });
  });

  describe("persistence debounce", () => {
    it("calls onPersist once after a burst of mutations", () => {
      const { store, persist } = makeStore();
      store.setEdgesForFile("id-src", [makeEdge("a", "A", 0.9)], 100);
      store.setEdgesForFile("id-src", [makeEdge("b", "B", 0.8)], 200);
      store.denyEdge("id-src", "c", "id-tgt");
      expect(persist).not.toHaveBeenCalled();
      jest.advanceTimersByTime(400);
      expect(persist).toHaveBeenCalledTimes(1);
    });
  });

  describe("shutdown", () => {
    it("flushes pending persist and resolves", async () => {
      const { store, persist } = makeStore();
      store.setEdgesForFile("id-src", [makeEdge("a", "A", 0.9)], 100);
      const shutdownPromise = store.shutdown();
      jest.runAllTimers();
      await shutdownPromise;
      expect(persist).toHaveBeenCalledTimes(1);
    });
  });

  describe("getMtime", () => {
    it("returns mtime stored with setEdgesForFile", () => {
      const { store } = makeStore();
      store.setEdgesForFile("id-src", [], 999);
      expect(store.getMtime("id-src")).toBe(999);
    });

    it("returns null for unknown id", () => {
      const { store } = makeStore();
      expect(store.getMtime("unknown")).toBeNull();
    });
  });

  describe("allSourceIds", () => {
    it("returns all ids that have been stored", () => {
      const { store } = makeStore();
      store.setEdgesForFile("id-a", [], 100);
      store.setEdgesForFile("id-b", [], 200);
      expect(store.allSourceIds().sort()).toEqual(["id-a", "id-b"]);
    });
  });
});
