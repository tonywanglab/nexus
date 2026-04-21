import { PersistedState, createEmptyPersistedState, DEFAULT_SIMILARITY_THRESHOLD } from "./types";

export function serialize(state: PersistedState): unknown {
  return state;
}

export function deserialize(raw: unknown): PersistedState {
  const defaults = createEmptyPersistedState();
  if (raw == null || typeof raw !== "object" || Array.isArray(raw)) return defaults;

  const obj = raw as Record<string, unknown>;
  if (obj["version"] !== 1) return defaults;

  return {
    version: 1,
    similarityThreshold:
      typeof obj["similarityThreshold"] === "number"
        ? obj["similarityThreshold"]
        : DEFAULT_SIMILARITY_THRESHOLD,
    notes:
      obj["notes"] != null && typeof obj["notes"] === "object" && !Array.isArray(obj["notes"])
        ? (obj["notes"] as PersistedState["notes"])
        : {},
    edges:
      obj["edges"] != null && typeof obj["edges"] === "object" && !Array.isArray(obj["edges"])
        ? (obj["edges"] as PersistedState["edges"])
        : {},
    denials: Array.isArray(obj["denials"]) ? (obj["denials"] as PersistedState["denials"]) : [],
    approvals: Array.isArray(obj["approvals"])
      ? (obj["approvals"] as PersistedState["approvals"])
      : [],
  };
}
