/**
 * Clean vault markdown files for evaluation:
 *
 * Pass 1 — Remove aliased wikilinks: [[Target|display]] → "" entirely.
 *          These cause FNs because the display text doesn't match the target title.
 *          Non-aliased wikilinks [[Target]] are kept.
 *
 * Pass 2 — Strip unlinked title mentions (same as before).
 *          After pass 1, new plain-text vault titles may be exposed; remove them.
 *
 * Preserves: YAML frontmatter, content after "%% wiki footer", a file's own title.
 */

import * as fs from "fs";
import * as path from "path";

const VAULT_DIR = path.resolve(__dirname, "../senior-thesis-vault");

// ── Collect all .md files and their titles ──────────────────────────

function walkDir(dir: string): string[] {
  const results: string[] = [];
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      if (entry.name === ".obsidian" || entry.name === "Assets") continue;
      results.push(...walkDir(full));
    } else if (entry.name.endsWith(".md")) {
      results.push(full);
    }
  }
  return results;
}

/** Split body from frontmatter and wiki footer (which we never touch). */
function splitProtectedZones(content: string): { frontmatter: string; body: string; footer: string } {
  let frontmatter = "";
  let body = content;
  if (content.startsWith("---")) {
    const endIdx = content.indexOf("\n---", 3);
    if (endIdx !== -1) {
      const fmEnd = endIdx + 4;
      frontmatter = content.slice(0, fmEnd);
      body = content.slice(fmEnd);
    }
  }

  let footer = "";
  const footerIdx = body.indexOf("%% wiki footer");
  if (footerIdx !== -1) {
    footer = body.slice(footerIdx);
    body = body.slice(0, footerIdx);
  }

  return { frontmatter, body, footer };
}

const allFiles = walkDir(VAULT_DIR);
const allTitles = allFiles
  .map((f) => path.basename(f, ".md"))
  .sort((a, b) => b.length - a.length); // longest first

console.log(`Found ${allFiles.length} files, ${allTitles.length} titles`);

// ── Pass 1: Remove aliased wikilinks ────────────────────────────────

let totalAliasesRemoved = 0;

console.log("\n── Pass 1: Remove aliased wikilinks ──");

for (const filePath of allFiles) {
  let content = fs.readFileSync(filePath, "utf-8");
  const { frontmatter, body, footer } = splitProtectedZones(content);

  let count = 0;
  // Match [[target|display]] but not embeds ![[...]]
  const cleaned = body.replace(/(?<!!)\[\[([^\]|]+)\|([^\]]+)\]\]/g, () => {
    count++;
    return "";
  });

  if (count > 0) {
    fs.writeFileSync(filePath, frontmatter + cleaned + footer, "utf-8");
    totalAliasesRemoved += count;
    console.log(`  ${path.relative(VAULT_DIR, filePath)}: removed ${count} aliased links`);
  }
}

console.log(`\nPass 1 done. Removed ${totalAliasesRemoved} aliased wikilinks.`);

// ── Pass 2: Strip unlinked title mentions ───────────────────────────

let totalStripped = 0;

console.log("\n── Pass 2: Strip unlinked title mentions ──");

for (const filePath of allFiles) {
  const ownTitle = path.basename(filePath, ".md");
  const titlesToStrip = allTitles.filter((t) => t !== ownTitle);
  if (titlesToStrip.length === 0) continue;

  let content = fs.readFileSync(filePath, "utf-8");
  const { frontmatter, body, footer } = splitProtectedZones(content);

  // Protect remaining wikilinks with placeholders
  const links: string[] = [];
  let work = body.replace(/\[\[[^\]]*\]\]/g, (match) => {
    links.push(match);
    return `\x00LINK_${links.length - 1}\x00`;
  });

  // Strip each title (longest first)
  let fileStripped = 0;
  for (const title of titlesToStrip) {
    const escaped = title.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    const re = new RegExp(`(?<![a-zA-Z0-9])${escaped}(?![a-zA-Z0-9])`, "gi");
    work = work.replace(re, () => {
      fileStripped++;
      return "";
    });
  }

  // Restore wikilinks
  work = work.replace(/\x00LINK_(\d+)\x00/g, (_, idx) => links[Number(idx)]);

  if (fileStripped > 0) {
    fs.writeFileSync(filePath, frontmatter + work + footer, "utf-8");
    totalStripped += fileStripped;
    console.log(`  ${path.relative(VAULT_DIR, filePath)}: stripped ${fileStripped} mentions`);
  }
}

console.log(`\nPass 2 done. Stripped ${totalStripped} unlinked title mentions.`);
console.log(`\nTotal: ${totalAliasesRemoved} aliased links removed, ${totalStripped} unlinked mentions stripped.`);
