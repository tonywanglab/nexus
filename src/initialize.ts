import { Vault, TFile } from "obsidian";
import { NoteRegistry } from "./note-registry";
import { EdgeStore } from "./edge-store";
import { JobQueue } from "./job-queue";

// diffs the current vault state against persisted cache.
// - Mints ids for files not yet tracked.
// - Enqueues files whose mtime has changed.
// - Drops orphaned ids (files no longer in the vault).
export async function initializeFromCache(
  vault: Vault,
  registry: NoteRegistry,
  store: EdgeStore,
  queue: JobQueue,
): Promise<{ enqueuedPaths: string[] }> {
  const mdFiles = vault.getMarkdownFiles() as TFile[];
  console.log(`Nexus: vault scan — ${mdFiles.length} markdown files`);

  const { added, reattached, dropped } = registry.reconcile(
    mdFiles.map((f) => ({ path: f.path, mtime: f.stat.mtime, size: f.stat.size })),
  );

  if (dropped.length) console.log(`Nexus: dropped ${dropped.length} orphaned note(s) from cache`);
  if (reattached.length) console.log(`Nexus: reattached ${reattached.length} renamed note(s) via size match`);
  if (added.length) console.log(`Nexus: registered ${added.length} new note(s)`);

  for (const id of dropped) {
    store.dropNote(id);
  }

  let skipped = 0;
  const enqueuedPaths: string[] = [];
  for (const file of mdFiles) {
    const id = registry.getId(file.path);
    if (!id) continue;
    const cachedMtime = store.getMtime(id);
    if (cachedMtime === null || cachedMtime !== file.stat.mtime) {
      queue.enqueue(file.path, "process");
      enqueuedPaths.push(file.path);
    } else {
      skipped++;
    }
  }

  console.log(`Nexus: cache init complete — ${skipped} up-to-date, ${enqueuedPaths.length} queued for processing`);
  return { enqueuedPaths };
}
