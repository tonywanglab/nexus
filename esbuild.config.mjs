import esbuild from "esbuild";
import process from "process";
import builtins from "builtin-modules";
import { existsSync, readFileSync, mkdirSync, copyFileSync } from "fs";
import { resolve, join } from "path";

const prod = process.argv[2] === "production";

// Read vault path from .env for dev mode
let vaultPluginDir = null;
if (!prod && existsSync(".env")) {
  const lines = readFileSync(".env", "utf-8").split("\n");
  for (const line of lines) {
    const match = line.match(/^VAULT_PATH=(.+)$/);
    if (match) {
      vaultPluginDir = resolve(match[1].trim(), ".obsidian", "plugins", "nexus");
    }
  }
}

const outfile = vaultPluginDir ? join(vaultPluginDir, "main.js") : "main.js";

// Ensure output directory exists when targeting vault
if (vaultPluginDir) {
  mkdirSync(vaultPluginDir, { recursive: true });
  // Copy manifest so Obsidian recognizes the plugin
  copyFileSync("manifest.json", join(vaultPluginDir, "manifest.json"));
}

const context = await esbuild.context({
  entryPoints: ["src/main.ts"],
  bundle: true,
  external: [
    "obsidian",
    "electron",
    "@codemirror/autocomplete",
    "@codemirror/collab",
    "@codemirror/commands",
    "@codemirror/language",
    "@codemirror/lint",
    "@codemirror/search",
    "@codemirror/state",
    "@codemirror/view",
    "@lezer/common",
    "@lezer/highlight",
    "@lezer/lr",
    ...builtins,
  ],
  format: "cjs",
  target: "es2018",
  logLevel: "info",
  sourcemap: prod ? false : "inline",
  treeShaking: true,
  outfile,
});

if (prod) {
  await context.rebuild();
  process.exit(0);
} else {
  await context.watch();
}
