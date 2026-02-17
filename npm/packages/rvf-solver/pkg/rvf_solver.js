/**
 * @ruvector/rvf-solver — JS glue for the RVF Solver WASM module.
 *
 * Loads the .wasm binary and re-exports all C-ABI functions plus the
 * WASM linear memory object.
 *
 * Works in Node.js (CJS). For browser/ESM, pass WASM bytes directly.
 */

var wasmInstance = null;

var _isNode = typeof process !== 'undefined' &&
  typeof process.versions !== 'undefined' &&
  typeof process.versions.node === 'string';

/**
 * Initialize the WASM module.
 * Returns the exports object with all rvf_solver_* functions and `memory`.
 *
 * @param {ArrayBuffer|BufferSource|WebAssembly.Module|string} [input]
 *   Optional pre-loaded bytes, Module, or file path override.
 */
async function init(input) {
  if (wasmInstance) return wasmInstance;

  var wasmBytes;

  if (input instanceof ArrayBuffer || ArrayBuffer.isView(input)) {
    wasmBytes = input;
  } else if (typeof WebAssembly !== 'undefined' && input instanceof WebAssembly.Module) {
    var inst = await WebAssembly.instantiate(input, {});
    wasmInstance = inst.exports;
    return wasmInstance;
  } else if (_isNode) {
    var fs = require('node:fs');
    var path = require('node:path');
    var wasmPath;
    if (typeof input === 'string') {
      wasmPath = input;
    } else {
      wasmPath = path.join(__dirname, 'rvf_solver_bg.wasm');
    }
    wasmBytes = fs.readFileSync(wasmPath);
  } else {
    // Browser: caller must provide bytes or Module
    throw new Error(
      '@ruvector/rvf-solver: In browser environments, pass WASM bytes or ' +
      'a WebAssembly.Module to init(). Example: ' +
      'init(await fetch("rvf_solver_bg.wasm").then(r => r.arrayBuffer()))'
    );
  }

  var compiled = await WebAssembly.instantiate(wasmBytes, {});
  wasmInstance = compiled.instance.exports;
  return wasmInstance;
}

// CJS export
init.default = init;
module.exports = init;
