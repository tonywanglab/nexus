import * as fs from "fs";
import * as readline from "readline";

/**
 * Async line iterator that does not use readline's Symbol.asyncIterator.
 * Node 20+ can throw ERR_USE_AFTER_CLOSE from that iterator when the consumer
 * awaits between lines (e.g. embedding batches), which can drop the final line.
 */
export async function* readLinesAsync(path: string): AsyncGenerator<string> {
  const rl = readline.createInterface({
    input: fs.createReadStream(path),
    crlfDelay: Infinity,
  });
  const queue: string[] = [];
  let resolveWait: (() => void) | null = null;
  let closed = false;
  let error: Error | undefined;

  const notify = (): void => {
    if (resolveWait) {
      resolveWait();
      resolveWait = null;
    }
  };

  rl.on("line", (line: string) => {
    queue.push(line);
    notify();
  });
  rl.on("close", () => {
    closed = true;
    notify();
  });
  rl.on("error", (err: Error) => {
    error = err;
    closed = true;
    notify();
  });

  try {
    while (true) {
      if (error) throw error;
      if (queue.length > 0) {
        yield queue.shift()!;
        continue;
      }
      if (closed) break;
      await new Promise<void>((resolve) => {
        resolveWait = resolve;
      });
    }
  } finally {
    rl.close();
  }
}
