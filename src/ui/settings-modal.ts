import { App, Modal } from "obsidian";
import { EdgeStore } from "../edge-store";
import { JobQueue } from "../job-queue";
import { NoteRegistry } from "../note-registry";
import { ConfirmModal } from "./confirm-modal";
import { ProgressToast } from "./progress-toast";

export class NexusSettingsModal extends Modal {
  private store: EdgeStore;
  private queue: JobQueue;
  private registry: NoteRegistry;

  constructor(app: App, store: EdgeStore, queue: JobQueue, registry: NoteRegistry) {
    super(app);
    this.store = store;
    this.queue = queue;
    this.registry = registry;
  }

  onOpen(): void {
    const { contentEl } = this;
    contentEl.createEl("h2", { text: "Nexus settings" });

    const row = contentEl.createEl("div", { cls: "nexus-setting-row" });
    row.createEl("label", { text: "Minimum similarity", cls: "nexus-setting-label" });

    const sliderWrap = row.createEl("div", { cls: "nexus-slider-wrap" });
    const slider = sliderWrap.createEl("input") as any;
    slider.setAttr("type", "range");
    slider.setAttr("min", "0");
    slider.setAttr("max", "1");
    slider.setAttr("step", "0.01");
    slider.setAttr("value", String(this.store.getThreshold()));

    const label = sliderWrap.createEl("span", {
      text: `${Math.round(this.store.getThreshold() * 100)}%`,
      cls: "nexus-slider-value",
    });

    let pendingValue = this.store.getThreshold();

    slider.addEventListener("input", () => {
      pendingValue = parseFloat(slider._attrs?.value ?? slider.value ?? "0");
      label.setText(`${Math.round(pendingValue * 100)}%`);
    });

    slider.addEventListener("change", async () => {
      const newVal = parseFloat(slider._attrs?.value ?? slider.value ?? "0");
      const oldVal = this.store.getThreshold();
      if (newVal === oldVal) return;

      if (newVal > oldVal) {
        this.store.setThresholdAndPrune(newVal);
      } else {
        const ids = this.store.allSourceIds();
        const minutes = Math.ceil((ids.length * 0.5) / 60);
        const confirmed = await ConfirmModal.prompt(
          this.app,
          `Lowering the threshold will re-resolve ${ids.length} file${ids.length !== 1 ? "s" : ""} (~${minutes} min). Continue?`,
        );
        if (!confirmed) {
          slider.setAttr("value", String(oldVal));
          label.setText(`${Math.round(oldVal * 100)}%`);
          return;
        }

        this.store.setThresholdNoPrune(newVal);
        const toast = new ProgressToast(ids.length);
        let processed = 0;
        const unsub = this.store.subscribe(() => {
          processed++;
          toast.tick();
          if (processed >= ids.length) unsub();
        });

        for (const id of ids) {
          const path = this.registry.getPath(id);
          if (path) this.queue.forceEnqueue(path, "process");
        }
      }
    });
  }

  onClose(): void {
    this.contentEl.empty();
  }
}
