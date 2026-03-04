import { Vault, TFile, TFolder } from "./__mocks__/obsidian";
import { EventListener, VaultEvent, VaultEventCallback } from "../event-listener";

function setup(cb?: VaultEventCallback) {
  const vault = new Vault();
  const callback = cb ?? jest.fn();
  const listener = new EventListener(vault as any, callback);
  return { vault, callback: callback as jest.Mock, listener };
}

describe("EventListener", () => {
  // ─── lifecycle ─────────────────────────────────────────────
  describe("subscribe / unsubscribe", () => {
    it("starts unsubscribed", () => {
      const { listener } = setup();
      expect(listener.isSubscribed).toBe(false);
    });

    it("subscribes to vault events", () => {
      const { listener } = setup();
      listener.subscribe();
      expect(listener.isSubscribed).toBe(true);
    });

    it("calling subscribe twice does not duplicate handlers", () => {
      const { vault, callback, listener } = setup();
      listener.subscribe();
      listener.subscribe();
      vault.trigger("create", new TFile("a.md"));
      expect(callback).toHaveBeenCalledTimes(1);
    });

    it("unsubscribes cleanly", () => {
      const { vault, callback, listener } = setup();
      listener.subscribe();
      listener.unsubscribe();
      expect(listener.isSubscribed).toBe(false);
      vault.trigger("create", new TFile("a.md"));
      expect(callback).not.toHaveBeenCalled();
    });

    it("can re-subscribe after unsubscribe", () => {
      const { vault, callback, listener } = setup();
      listener.subscribe();
      listener.unsubscribe();
      listener.subscribe();
      vault.trigger("modify", new TFile("a.md"));
      expect(callback).toHaveBeenCalledTimes(1);
    });
  });

  // ─── event routing ────────────────────────────────────────
  describe("event types", () => {
    it("emits create events", () => {
      const { vault, callback, listener } = setup();
      listener.subscribe();
      const file = new TFile("notes/hello.md");
      vault.trigger("create", file);
      expect(callback).toHaveBeenCalledWith<[VaultEvent]>({
        type: "create",
        file: file as any,
        oldPath: undefined,
      });
    });

    it("emits modify events", () => {
      const { vault, callback, listener } = setup();
      listener.subscribe();
      const file = new TFile("notes/hello.md");
      vault.trigger("modify", file);
      expect(callback).toHaveBeenCalledWith<[VaultEvent]>({
        type: "modify",
        file: file as any,
        oldPath: undefined,
      });
    });

    it("emits delete events", () => {
      const { vault, callback, listener } = setup();
      listener.subscribe();
      const file = new TFile("notes/hello.md");
      vault.trigger("delete", file);
      expect(callback).toHaveBeenCalledWith<[VaultEvent]>({
        type: "delete",
        file: file as any,
        oldPath: undefined,
      });
    });

    it("emits rename events with oldPath", () => {
      const { vault, callback, listener } = setup();
      listener.subscribe();
      const file = new TFile("notes/renamed.md");
      vault.trigger("rename", file, "notes/old-name.md");
      expect(callback).toHaveBeenCalledWith<[VaultEvent]>({
        type: "rename",
        file: file as any,
        oldPath: "notes/old-name.md",
      });
    });
  });

  // ─── filtering ────────────────────────────────────────────
  describe("file filtering", () => {
    it("ignores non-markdown files", () => {
      const { vault, callback, listener } = setup();
      listener.subscribe();
      vault.trigger("create", new TFile("image.png"));
      vault.trigger("modify", new TFile("data.json"));
      vault.trigger("delete", new TFile("style.css"));
      expect(callback).not.toHaveBeenCalled();
    });

    it("ignores folder events", () => {
      const { vault, callback, listener } = setup();
      listener.subscribe();
      vault.trigger("create", new TFolder("notes"));
      vault.trigger("delete", new TFolder("archive"));
      expect(callback).not.toHaveBeenCalled();
    });

    it("processes .md files from subdirectories", () => {
      const { vault, callback, listener } = setup();
      listener.subscribe();
      vault.trigger("create", new TFile("deep/nested/path/note.md"));
      expect(callback).toHaveBeenCalledTimes(1);
    });
  });

  // ─── multiple events ─────────────────────────────────────
  describe("multiple events", () => {
    it("handles a sequence of different events", () => {
      const { vault, callback, listener } = setup();
      listener.subscribe();
      vault.trigger("create", new TFile("a.md"));
      vault.trigger("modify", new TFile("b.md"));
      vault.trigger("delete", new TFile("c.md"));
      vault.trigger("rename", new TFile("d.md"), "old-d.md");
      expect(callback).toHaveBeenCalledTimes(4);
      expect(callback.mock.calls[0][0].type).toBe("create");
      expect(callback.mock.calls[1][0].type).toBe("modify");
      expect(callback.mock.calls[2][0].type).toBe("delete");
      expect(callback.mock.calls[3][0].type).toBe("rename");
    });

    it("handles rapid events on the same file", () => {
      const { vault, callback, listener } = setup();
      listener.subscribe();
      const file = new TFile("note.md");
      vault.trigger("modify", file);
      vault.trigger("modify", file);
      vault.trigger("modify", file);
      expect(callback).toHaveBeenCalledTimes(3);
    });
  });

  // ─── post-unsubscribe safety ──────────────────────────────
  describe("after unsubscribe", () => {
    it("no events fire for any event type", () => {
      const { vault, callback, listener } = setup();
      listener.subscribe();
      listener.unsubscribe();
      vault.trigger("create", new TFile("a.md"));
      vault.trigger("modify", new TFile("b.md"));
      vault.trigger("delete", new TFile("c.md"));
      vault.trigger("rename", new TFile("d.md"), "old.md");
      expect(callback).not.toHaveBeenCalled();
    });
  });
});
