// shims for @tensorflow/tfjs-node on modern Node (18+).
//
// tfjs-node 4.22 still calls `util.isNullOrUndefined`, which was removed in
// node 18+. Polyfill it before tfjs-node loads so its kernel wrappers don't
// crash on the first operation.
//
// import this module BEFORE any `@tensorflow/tfjs-node` import.

// eslint-disable-next-line @typescript-eslint/no-var-requires
const util = require("util") as { isNullOrUndefined?: (v: unknown) => boolean };
if (typeof util.isNullOrUndefined !== "function") {
  util.isNullOrUndefined = (v: unknown): boolean => v === null || v === undefined;
}
