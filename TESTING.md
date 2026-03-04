# Testing Nexus in Obsidian

## Setup (Automated)

✓ Dev watch is running (`npm run dev`)
✓ Plugin deployed to `/Users/tonywang/Documents/tony's vault/.obsidian/plugins/nexus/`

## Manual Steps in Obsidian

1. **Open Obsidian** (already done)
2. **Reload vault** (clears plugin cache):
   - Settings → About → Click "Reload app" button OR
   - Press **Cmd+R** in Obsidian
3. **Enable the plugin**:
   - Settings → Community plugins → Browse (or scroll to "Nexus")
   - Click "Enable" next to Nexus
4. **Open DevTools** to see console output:
   - Press **Cmd+Option+I** to open DevTools
   - Go to **Console** tab
   - You should see: `Nexus: loading plugin`

## Test It

### Test 1: Create a new note
1. In Obsidian, create a new note (Cmd+N)
2. Type some content
3. In DevTools Console, watch for:
   ```
   Nexus: processing notes/Untitled.md (high)
   ```
   This shows the job queue is working (active file = high priority = 200ms debounce)

### Test 2: Modify an existing note
1. Open any existing note
2. Make a small change (add a word)
3. Wait ~200ms
4. Check console for debounce message

### Test 3: Delete a note
1. Create a temporary test note
2. Delete it (or move to trash)
3. Console should show the delete event was captured (no processing fires, just cancellation)

### Test 4: Rapid edits → debounce
1. Create a note
2. Type several words rapidly
3. Console should show **only one** "processing" message after you stop typing
4. Even though you typed 10+ times, the queue debounced to a single job

### Test 5: Priority boost
1. Create note A and note B
2. Edit note A (it becomes active file → high priority = 200ms)
3. Quickly switch focus away and edit note B (normal priority = 500ms)
4. You should see note A process in ~200ms, note B in ~500ms

## Troubleshooting

### Plugin doesn't load?
- Check `/Users/tonywang/Documents/tony's vault/.obsidian/plugins/nexus/main.js` exists
- DevTools → Console should show `Nexus: loading plugin`
- Try Settings → Community plugins → turn off Restricted mode, reload, re-enable

### No console messages?
- Make sure DevTools console is focused (click Console tab)
- Try creating a new note to trigger events
- Check that Nexus is listed as "Enabled" in Settings → Community plugins

### Want to see more details?
- Edit `src/main.ts` and add more `console.log()` statements
- Save (esbuild rebuilds automatically)
- Reload Obsidian (Cmd+R)
- New logs will appear in console

## What's Next?

Once you confirm the event listener + job queue are working (you see console messages), the next phases are:
1. **YAKE-lite keyphrase extraction** — parse note content
2. **LCS-based alias resolution** — match keyphrases to existing note titles
3. **Stochastic methods** — embeddings via Ollama
4. **UI Modal** — approve/reject candidate links
