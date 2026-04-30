/**
 * Shared vault-loading helpers used by eval and benchmark scripts.
 */

import * as fs from "fs";
import * as path from "path";

export interface VaultFile {
  path: string;
  basename: string;
  content: string;
}

/** Recursively read every .md file under `vaultDir`. */
export function loadVault(vaultDir: string): VaultFile[] {
  const files: VaultFile[] = [];
  function walk(dir: string): void {
    for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
      const fullPath = path.join(dir, entry.name);
      if (entry.isDirectory()) {
        walk(fullPath);
      } else if (entry.isFile() && entry.name.endsWith(".md")) {
        files.push({
          path: fullPath,
          basename: entry.name.replace(/\.md$/, ""),
          content: fs.readFileSync(fullPath, "utf-8"),
        });
      }
    }
  }
  walk(vaultDir);
  return files;
}
