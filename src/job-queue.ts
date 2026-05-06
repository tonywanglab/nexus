import { JobType, QueueJob } from "./types";

export type ProcessCallback = (job: QueueJob) => void | Promise<void>;

export interface JobQueueOptions {
  debounceMs?: number;
  priorityDebounceMs?: number;
  cooldownMs?: number;
}

interface PendingEntry {
  timer: ReturnType<typeof setTimeout>;
  job: QueueJob;
}

export class JobQueue {
  private pending = new Map<string, PendingEntry>();
  private lastProcessed = new Map<string, number>();
  private activeFile: string | null = null;
  private readonly debounceMs: number;
  private readonly priorityDebounceMs: number;
  private readonly cooldownMs: number;
  /**
   * Serialization: when onProcess is async, at most one job runs at a time.
   * `serialRunning` is true while an async job is in flight; subsequent jobs
   * are held in `serialQueue` and started when the in-flight job settles.
   * Sync onProcess (unit-test mocks) leaves serialRunning=false so tests
   * don't have to flush microtasks after fake-timer advances.
   */
  private serialRunning = false;
  private serialQueue: QueueJob[] = [];

  constructor(
    private onProcess: ProcessCallback,
    options: JobQueueOptions = {}
  ) {
    this.debounceMs = options.debounceMs ?? 500;
    this.priorityDebounceMs = options.priorityDebounceMs ?? 200;
    this.cooldownMs = options.cooldownMs ?? 2000;
  }

  enqueue(filePath: string, type: JobType): void {
    const existing = this.pending.get(filePath);
    if (existing) {
      clearTimeout(existing.timer);
    }

    const isActive = filePath === this.activeFile;
    const delay = isActive ? this.priorityDebounceMs : this.debounceMs;

    const job: QueueJob = {
      filePath,
      type,
      priority: isActive ? "high" : "normal",
      enqueuedAt: Date.now(),
    };

    const timer = setTimeout(() => {
      this.pending.delete(filePath);
      const last = this.lastProcessed.get(filePath) ?? 0;
      if (Date.now() - last < this.cooldownMs) return;
      this.lastProcessed.set(filePath, Date.now());
      this.dispatch(job);
    }, delay);

    this.pending.set(filePath, { timer, job });
  }

  /** Like enqueue but bypasses the cooldown gate — for settings-triggered bulk reprocess. */
  forceEnqueue(filePath: string, type: JobType): void {
    const existing = this.pending.get(filePath);
    if (existing) clearTimeout(existing.timer);

    const isActive = filePath === this.activeFile;
    const delay = isActive ? this.priorityDebounceMs : this.debounceMs;

    const job: QueueJob = {
      filePath,
      type,
      priority: isActive ? "high" : "normal",
      enqueuedAt: Date.now(),
    };

    const timer = setTimeout(() => {
      this.pending.delete(filePath);
      this.lastProcessed.set(filePath, Date.now());
      this.dispatch(job); // no cooldown check
    }, delay);

    this.pending.set(filePath, { timer, job });
  }

  cancel(filePath: string): void {
    const entry = this.pending.get(filePath);
    if (entry) {
      clearTimeout(entry.timer);
      this.pending.delete(filePath);
    }
  }

  setActiveFile(filePath: string | null): void {
    this.activeFile = filePath;
  }

  flush(): void {
    const entries = [...this.pending.values()];
    for (const { timer } of entries) clearTimeout(timer);
    this.pending.clear();
    // Cooldown is only for timer-driven `enqueue`; flush must not drop jobs.
    for (const { job } of entries) {
      this.lastProcessed.set(job.filePath, Date.now());
      this.dispatch(job);
    }
  }

  /** Cancel all pending normal-priority jobs except for the given file. */
  cancelNormalExcept(filePath: string): void {
    for (const [path, entry] of this.pending) {
      if (path !== filePath && entry.job.priority === "normal") {
        clearTimeout(entry.timer);
        this.pending.delete(path);
      }
    }
  }

  /**
   * Run a job, serializing against any currently in-flight async job.
   * Sync onProcess calls (e.g. test mocks) bypass the queue entirely so
   * fake-timer tests don't need to flush microtasks.
   */
  private dispatch(job: QueueJob): void {
    if (this.serialRunning) {
      this.serialQueue.push(job);
      return;
    }
    let result: void | Promise<void>;
    try {
      result = this.onProcess(job);
    } catch {
      return;
    }
    if (result instanceof Promise) {
      this.serialRunning = true;
      result.then(() => this.afterDispatch(), () => this.afterDispatch());
    }
    // Sync result: serialRunning stays false — next dispatch starts immediately.
  }

  private afterDispatch(): void {
    while (true) {
      const next = this.serialQueue.shift();
      if (!next) {
        this.serialRunning = false;
        return;
      }
      let result: void | Promise<void>;
      try {
        result = this.onProcess(next);
      } catch {
        continue; // skip failed job, drain queue
      }
      if (result instanceof Promise) {
        result.then(() => this.afterDispatch(), () => this.afterDispatch());
        return; // still in-flight; afterDispatch will be called when done
      }
      // Sync result: loop to drain more jobs without microtask overhead
    }
  }

  clear(): void {
    for (const { timer } of this.pending.values()) {
      clearTimeout(timer);
    }
    this.pending.clear();
  }

  get pendingCount(): number {
    return this.pending.size;
  }

  hasPending(filePath: string): boolean {
    return this.pending.has(filePath);
  }
}
