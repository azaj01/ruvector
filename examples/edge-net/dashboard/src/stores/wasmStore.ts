import { create } from 'zustand';
import type { WASMModule, WASMBenchmark } from '../types';

interface WASMState {
  modules: WASMModule[];
  benchmarks: WASMBenchmark[];
  isInitialized: boolean;
  isLoading: boolean;
  error: string | null;

  // Actions
  setModules: (modules: WASMModule[]) => void;
  updateModule: (moduleId: string, updates: Partial<WASMModule>) => void;
  addBenchmark: (benchmark: WASMBenchmark) => void;
  clearBenchmarks: () => void;
  setInitialized: (initialized: boolean) => void;
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
  loadModule: (moduleId: string) => Promise<void>;
  runBenchmark: (moduleId: string) => Promise<WASMBenchmark | null>;
}

const defaultModules: WASMModule[] = [
  {
    id: 'edge-net',
    name: '@ruvector/edge-net',
    version: '0.1.1',
    loaded: false,
    size: 3200000,
    features: ['Time Crystal', 'DAG Attention', 'P2P Swarm', 'Credit Economy'],
    status: 'unloaded',
  },
  {
    id: 'attention-unified',
    name: '@ruvector/attention-unified-wasm',
    version: '0.1.0',
    loaded: false,
    size: 850000,
    features: ['DAG Attention', 'Critical Path', 'Topological Sort'],
    status: 'unloaded',
  },
  {
    id: 'economy',
    name: '@ruvector/economy-wasm',
    version: '0.1.0',
    loaded: false,
    size: 620000,
    features: ['Credit Marketplace', 'Staking', 'Governance'],
    status: 'unloaded',
  },
  {
    id: 'exotic',
    name: '@ruvector/exotic-wasm',
    version: '0.1.0',
    loaded: false,
    size: 780000,
    features: ['Exotic AI', 'MinCut Signals', 'RAC Coherence'],
    status: 'unloaded',
  },
  {
    id: 'learning',
    name: '@ruvector/learning-wasm',
    version: '0.1.0',
    loaded: false,
    size: 540000,
    features: ['Q-Learning', 'Pattern Recognition', 'Self-Improvement'],
    status: 'unloaded',
  },
  {
    id: 'nervous-system',
    name: '@ruvector/nervous-system-wasm',
    version: '0.1.0',
    loaded: false,
    size: 920000,
    features: ['Neural Coordination', 'Homeostasis', 'Reflex Arcs'],
    status: 'unloaded',
  },
];

export const useWASMStore = create<WASMState>((set, get) => ({
  modules: defaultModules,
  benchmarks: [],
  isInitialized: false,
  isLoading: false,
  error: null,

  setModules: (modules) => set({ modules }),

  updateModule: (moduleId, updates) =>
    set((state) => ({
      modules: state.modules.map((m) =>
        m.id === moduleId ? { ...m, ...updates } : m
      ),
    })),

  addBenchmark: (benchmark) =>
    set((state) => ({
      benchmarks: [...state.benchmarks, benchmark],
    })),

  clearBenchmarks: () => set({ benchmarks: [] }),

  setInitialized: (initialized) => set({ isInitialized: initialized }),
  setLoading: (loading) => set({ isLoading: loading }),
  setError: (error) => set({ error }),

  loadModule: async (moduleId) => {
    const { updateModule } = get();

    updateModule(moduleId, { status: 'loading' });

    try {
      // Simulate WASM loading
      await new Promise((resolve) => setTimeout(resolve, 1000 + Math.random() * 1000));

      updateModule(moduleId, {
        status: 'ready',
        loaded: true,
      });

      console.log(`[WASM] Module ${moduleId} loaded successfully`);
    } catch (error) {
      updateModule(moduleId, {
        status: 'error',
        error: error instanceof Error ? error.message : 'Unknown error',
      });
    }
  },

  runBenchmark: async (moduleId) => {
    const { modules, addBenchmark } = get();
    const module = modules.find((m) => m.id === moduleId);

    if (!module || !module.loaded) {
      console.warn(`[WASM] Cannot benchmark unloaded module: ${moduleId}`);
      return null;
    }

    // Simulate benchmark
    const iterations = 1000;
    const times: number[] = [];

    for (let i = 0; i < iterations; i++) {
      const start = performance.now();
      // Simulate operation
      await new Promise((r) => setTimeout(r, 0));
      times.push(performance.now() - start);
    }

    const benchmark: WASMBenchmark = {
      moduleId,
      operation: 'default',
      iterations,
      avgTime: times.reduce((a, b) => a + b, 0) / times.length,
      minTime: Math.min(...times),
      maxTime: Math.max(...times),
      throughput: iterations / (times.reduce((a, b) => a + b, 0) / 1000),
    };

    addBenchmark(benchmark);
    console.log(`[WASM] Benchmark for ${moduleId}:`, benchmark);

    return benchmark;
  },
}));
