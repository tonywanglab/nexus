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
      this.onProcess(job);
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
    for (const { timer } of entries) {
      clearTimeout(timer);
    }
    this.pending.clear();
    for (const { job } of entries) {
      this.onProcess(job);
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
