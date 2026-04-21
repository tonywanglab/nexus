import { serialize, deserialize } from "../persistence";
import { PersistedState, createEmptyPersistedState, CandidateEdge } from "../types";

function sampleState(): PersistedState {
  const phrase = {
    phrase: "neural nets",
    score: 0.2,
    startOffset: 0,
    endOffset: 11,
    spanId: "0:11",
  };
  const edge: CandidateEdge = {
    sourcePath: "a.md",
    sourceId: "id-a",
    phrase,
    targetPath: "Neural Networks",
    targetId: "id-b",
    similarity: 0.92,
    matchType: "deterministic",
  };
  return {
    version: 1,
    similarityThreshold: 0.75,
    notes: {
      "id-a": { path: "a.md", mtime: 100 },
      "id-b": { path: "Neural Networks.md", mtime: 200 },
    },
    edges: { "id-a": [edge] },
    denials: [{ sourceId: "id-a", phraseKey: "spam", targetId: "id-b" }],
    approvals: [
      { sourceId: "id-a", phraseKey: "neural nets", targetId: "id-b", approvedAt: 1700 },
    ],
  };
}

describe("persistence", () => {
  describe("round-trip", () => {
    it("serialize → deserialize preserves state", () => {
      const s = sampleState();
      const restored = deserialize(JSON.parse(JSON.stringify(serialize(s))));
      expect(restored).toEqual(s);
    });

    it("empty state round-trips", () => {
      const s = createEmptyPersistedState();
      const restored = deserialize(JSON.parse(JSON.stringify(serialize(s))));
      expect(restored).toEqual(s);
    });
  });

  describe("deserialize robustness", () => {
    it("null input returns default empty state", () => {
      expect(deserialize(null)).toEqual(createEmptyPersistedState());
    });

    it("undefined input returns default empty state", () => {
      expect(deserialize(undefined)).toEqual(createEmptyPersistedState());
    });

    it("unknown version returns default empty state", () => {
      expect(deserialize({ version: 999, notes: {} })).toEqual(createEmptyPersistedState());
    });

    it("missing required fields fills with defaults", () => {
      const restored = deserialize({ version: 1 });
      expect(restored.notes).toEqual({});
      expect(restored.edges).toEqual({});
      expect(restored.denials).toEqual([]);
      expect(restored.approvals).toEqual([]);
      expect(typeof restored.similarityThreshold).toBe("number");
    });

    it("malformed shape for a subfield does not throw", () => {
      const restored = deserialize({
        version: 1,
        notes: "not an object",
        edges: [],
        denials: "nope",
      });
      expect(restored.notes).toEqual({});
      expect(restored.edges).toEqual({});
      expect(restored.denials).toEqual([]);
    });
  });
});
