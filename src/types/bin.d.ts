// ambient declaration so TypeScript accepts raw `.bin` imports bundled by
// esbuild's `binary` loader as a default-exported Uint8Array.
declare module "*.bin" {
  const data: Uint8Array;
  export default data;
}
