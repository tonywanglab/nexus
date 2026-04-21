import { JobQueue, ProcessCallback } from "../job-queue";
import { QueueJob } from "../types";

function setup(opts?: { debounceMs?: number; priorityDebounceMs?: number; cooldownMs?: number }) {
  const onProcess: jest.Mock<void, [QueueJob]> = jest.fn();
  const queue = new JobQueue(onProcess as ProcessCallback, {
    debounceMs: opts?.debounceMs ?? 500,
    priorityDebounceMs: opts?.priorityDebounceMs ?? 200,
    cooldownMs: opts?.cooldownMs ?? 2000,
  });
  return { onProcess, queue };
}

describe("JobQueue", () => {
  beforeEach(() => jest.useFakeTimers());
  afterEach(() => jest.useRealTimers());

  // ─── basic debouncing ─────────────────────────────────────
  describe("debouncing", () => {
    it("fires a single job after the debounce delay", () => {
      const { onProcess, queue } = setup();
      queue.enqueue("a.md", "process");
      expect(onProcess).not.toHaveBeenCalled();
      jest.advanceTimersByTime(500);
      expect(onProcess).toHaveBeenCalledTimes(1);
      expect(onProcess).toHaveBeenCalledWith(
        expect.objectContaining({ filePath: "a.md", type: "process" })
      );
    });

    it("does not fire before the full delay", () => {
      const { onProcess, queue } = setup();
      queue.enqueue("a.md", "process");
      jest.advanceTimersByTime(499);
      expect(onProcess).not.toHaveBeenCalled();
    });

    it("resets the timer on repeated enqueues for the same file", () => {
      const { onProcess, queue } = setup();
      queue.enqueue("a.md", "process");
      jest.advanceTimersByTime(300);
      queue.enqueue("a.md", "process"); // resets timer
      jest.advanceTimersByTime(300);
      expect(onProcess).not.toHaveBeenCalled(); // only 300ms since reset
      jest.advanceTimersByTime(200);
      expect(onProcess).toHaveBeenCalledTimes(1);
    });

    it("uses the latest job type when debounced", () => {
      const { onProcess, queue } = setup();
      queue.enqueue("a.md", "process");
      jest.advanceTimersByTime(100);
      queue.enqueue("a.md", "reindex");
      jest.advanceTimersByTime(500);
      expect(onProcess).toHaveBeenCalledWith(
        expect.objectContaining({ filePath: "a.md", type: "reindex" })
      );
    });
  });

  // ─── per-file isolation ───────────────────────────────────
  describe("per-file isolation", () => {
    it("debounces files independently", () => {
      const { onProcess, queue } = setup();
      queue.enqueue("a.md", "process");
      jest.advanceTimersByTime(250);
      queue.enqueue("b.md", "process");
      jest.advanceTimersByTime(250);
      // a has waited 500ms total → fires; b only 250ms → not yet
      expect(onProcess).toHaveBeenCalledTimes(1);
      expect(onProcess).toHaveBeenCalledWith(
        expect.objectContaining({ filePath: "a.md" })
      );
      jest.advanceTimersByTime(250);
      expect(onProcess).toHaveBeenCalledTimes(2);
      expect(onProcess).toHaveBeenLastCalledWith(
        expect.objectContaining({ filePath: "b.md" })
      );
    });

    it("resetting one file does not affect another", () => {
      const { onProcess, queue } = setup();
      queue.enqueue("a.md", "process");
      queue.enqueue("b.md", "process");
      jest.advanceTimersByTime(300);
      queue.enqueue("a.md", "process"); // reset only a
      jest.advanceTimersByTime(200);
      // b fires at 500ms total; a was reset at 300ms so needs 500 more from reset
      expect(onProcess).toHaveBeenCalledTimes(1);
      expect(onProcess).toHaveBeenCalledWith(
        expect.objectContaining({ filePath: "b.md" })
      );
    });
  });

  // ─── priority / active file ───────────────────────────────
  describe("active file priority", () => {
    it("active file uses the shorter priority debounce", () => {
      const { onProcess, queue } = setup();
      queue.setActiveFile("a.md");
      queue.enqueue("a.md", "process");
      jest.advanceTimersByTime(200);
      expect(onProcess).toHaveBeenCalledTimes(1);
    });

    it("active file job has priority='high'", () => {
      const { onProcess, queue } = setup();
      queue.setActiveFile("a.md");
      queue.enqueue("a.md", "process");
      jest.advanceTimersByTime(200);
      expect(onProcess).toHaveBeenCalledWith(
        expect.objectContaining({ priority: "high" })
      );
    });

    it("non-active file has priority='normal' and uses full debounce", () => {
      const { onProcess, queue } = setup();
      queue.setActiveFile("a.md");
      queue.enqueue("b.md", "process");
      jest.advanceTimersByTime(200);
      expect(onProcess).not.toHaveBeenCalled();
      jest.advanceTimersByTime(300);
      expect(onProcess).toHaveBeenCalledWith(
        expect.objectContaining({ filePath: "b.md", priority: "normal" })
      );
    });

    it("changing active file affects subsequent enqueues", () => {
      const { onProcess, queue } = setup();
      queue.setActiveFile("a.md");
      queue.enqueue("a.md", "process");
      jest.advanceTimersByTime(200);
      expect(onProcess).toHaveBeenCalledTimes(1);

      queue.setActiveFile("b.md");
      queue.enqueue("b.md", "process");
      jest.advanceTimersByTime(200);
      expect(onProcess).toHaveBeenCalledTimes(2);
    });

    it("clearing active file (null) reverts to normal debounce", () => {
      const { onProcess, queue } = setup();
      queue.setActiveFile("a.md");
      queue.setActiveFile(null);
      queue.enqueue("a.md", "process");
      jest.advanceTimersByTime(200);
      expect(onProcess).not.toHaveBeenCalled();
      jest.advanceTimersByTime(300);
      expect(onProcess).toHaveBeenCalledTimes(1);
    });
  });

  // ─── cancellation ────────────────────────────────────────
  describe("cancel", () => {
    it("cancels a pending job so it never fires", () => {
      const { onProcess, queue } = setup();
      queue.enqueue("a.md", "process");
      queue.cancel("a.md");
      jest.advanceTimersByTime(1000);
      expect(onProcess).not.toHaveBeenCalled();
    });

    it("cancelling a non-existent file is a no-op", () => {
      const { queue } = setup();
      expect(() => queue.cancel("nope.md")).not.toThrow();
    });

    it("does not affect other pending jobs", () => {
      const { onProcess, queue } = setup();
      queue.enqueue("a.md", "process");
      queue.enqueue("b.md", "process");
      queue.cancel("a.md");
      jest.advanceTimersByTime(500);
      expect(onProcess).toHaveBeenCalledTimes(1);
      expect(onProcess).toHaveBeenCalledWith(
        expect.objectContaining({ filePath: "b.md" })
      );
    });
  });

  // ─── flush ────────────────────────────────────────────────
  describe("flush", () => {
    it("immediately processes all pending jobs", () => {
      const { onProcess, queue } = setup();
      queue.enqueue("a.md", "process");
      queue.enqueue("b.md", "process");
      queue.flush();
      expect(onProcess).toHaveBeenCalledTimes(2);
    });

    it("clears pending count to zero", () => {
      const { queue } = setup();
      queue.enqueue("a.md", "process");
      queue.enqueue("b.md", "process");
      queue.flush();
      expect(queue.pendingCount).toBe(0);
    });

    it("does not double-fire after flush when timers advance", () => {
      const { onProcess, queue } = setup();
      queue.enqueue("a.md", "process");
      queue.flush();
      jest.advanceTimersByTime(1000);
      expect(onProcess).toHaveBeenCalledTimes(1);
    });

    it("flush on empty queue is a no-op", () => {
      const { onProcess, queue } = setup();
      queue.flush();
      expect(onProcess).not.toHaveBeenCalled();
    });

    it("flush runs pending jobs even when the same path is under cooldown", () => {
      const { onProcess, queue } = setup({ cooldownMs: 2000 });
      queue.enqueue("a.md", "process");
      jest.advanceTimersByTime(500);
      expect(onProcess).toHaveBeenCalledTimes(1);
      queue.enqueue("a.md", "reindex");
      queue.flush();
      expect(onProcess).toHaveBeenCalledTimes(2);
      expect(onProcess).toHaveBeenLastCalledWith(
        expect.objectContaining({ filePath: "a.md", type: "reindex" })
      );
    });
  });

  // ─── clear ────────────────────────────────────────────────
  describe("clear", () => {
    it("discards all pending jobs without processing", () => {
      const { onProcess, queue } = setup();
      queue.enqueue("a.md", "process");
      queue.enqueue("b.md", "process");
      queue.clear();
      jest.advanceTimersByTime(1000);
      expect(onProcess).not.toHaveBeenCalled();
      expect(queue.pendingCount).toBe(0);
    });
  });

  // ─── pendingCount / hasPending ────────────────────────────
  describe("state queries", () => {
    it("pendingCount starts at 0", () => {
      const { queue } = setup();
      expect(queue.pendingCount).toBe(0);
    });

    it("pendingCount reflects enqueued jobs", () => {
      const { queue } = setup();
      queue.enqueue("a.md", "process");
      queue.enqueue("b.md", "process");
      expect(queue.pendingCount).toBe(2);
    });

    it("pendingCount decrements when a job fires", () => {
      const { queue } = setup();
      queue.enqueue("a.md", "process");
      jest.advanceTimersByTime(500);
      expect(queue.pendingCount).toBe(0);
    });

    it("pendingCount decrements on cancel", () => {
      const { queue } = setup();
      queue.enqueue("a.md", "process");
      queue.enqueue("b.md", "process");
      queue.cancel("a.md");
      expect(queue.pendingCount).toBe(1);
    });

    it("hasPending returns true for queued file", () => {
      const { queue } = setup();
      queue.enqueue("a.md", "process");
      expect(queue.hasPending("a.md")).toBe(true);
      expect(queue.hasPending("b.md")).toBe(false);
    });

    it("hasPending returns false after job fires", () => {
      const { queue } = setup();
      queue.enqueue("a.md", "process");
      jest.advanceTimersByTime(500);
      expect(queue.hasPending("a.md")).toBe(false);
    });
  });

  // ─── integration: event listener → job queue ──────────────
  describe("integration with event listener pattern", () => {
    it("simulates create → modify → modify debounce cycle", () => {
      const { onProcess, queue } = setup();
      // File created
      queue.enqueue("new-note.md", "process");
      jest.advanceTimersByTime(100);
      // User starts typing (modify)
      queue.enqueue("new-note.md", "process");
      jest.advanceTimersByTime(100);
      // Still typing
      queue.enqueue("new-note.md", "process");
      jest.advanceTimersByTime(500);
      // Only one final process fires
      expect(onProcess).toHaveBeenCalledTimes(1);
    });

    it("simulates rename: cancel old path, enqueue new path", () => {
      const { onProcess, queue } = setup();
      queue.enqueue("old-name.md", "process");
      jest.advanceTimersByTime(100);
      // Rename event: cancel old, enqueue new
      queue.cancel("old-name.md");
      queue.enqueue("new-name.md", "reindex");
      jest.advanceTimersByTime(500);
      expect(onProcess).toHaveBeenCalledTimes(1);
      expect(onProcess).toHaveBeenCalledWith(
        expect.objectContaining({ filePath: "new-name.md", type: "reindex" })
      );
    });

    it("simulates delete cancelling a pending process", () => {
      const { onProcess, queue } = setup();
      queue.enqueue("doomed.md", "process");
      jest.advanceTimersByTime(100);
      queue.cancel("doomed.md");
      jest.advanceTimersByTime(500);
      expect(onProcess).not.toHaveBeenCalled();
    });
  });

  // ─── forceEnqueue ──────────────────────────────────────────
  describe("forceEnqueue", () => {
    it("processes even when within cooldown window", () => {
      const { onProcess, queue } = setup();
      queue.enqueue("a.md", "process");
      jest.advanceTimersByTime(600);
      expect(onProcess).toHaveBeenCalledTimes(1);
      // Normal enqueue within cooldown is silently dropped.
      queue.enqueue("a.md", "process");
      jest.advanceTimersByTime(600);
      expect(onProcess).toHaveBeenCalledTimes(1);
      // forceEnqueue bypasses cooldown.
      queue.forceEnqueue("a.md", "process");
      jest.advanceTimersByTime(600);
      expect(onProcess).toHaveBeenCalledTimes(2);
    });
  });
});
