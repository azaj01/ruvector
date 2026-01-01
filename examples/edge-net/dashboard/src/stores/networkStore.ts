import { create } from 'zustand';
import type { NetworkStats, NodeInfo, TimeCrystal, CreditBalance } from '../types';

interface NetworkState {
  stats: NetworkStats;
  nodes: NodeInfo[];
  timeCrystal: TimeCrystal;
  credits: CreditBalance;
  isConnected: boolean;
  isLoading: boolean;
  error: string | null;

  // Actions
  setStats: (stats: Partial<NetworkStats>) => void;
  addNode: (node: NodeInfo) => void;
  removeNode: (nodeId: string) => void;
  updateNode: (nodeId: string, updates: Partial<NodeInfo>) => void;
  setTimeCrystal: (crystal: Partial<TimeCrystal>) => void;
  setCredits: (credits: Partial<CreditBalance>) => void;
  setConnected: (connected: boolean) => void;
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
  simulateActivity: () => void;
}

const initialStats: NetworkStats = {
  totalNodes: 1247,
  activeNodes: 892,
  totalCompute: 45.8,
  creditsEarned: 12847,
  tasksCompleted: 89432,
  uptime: 99.7,
  latency: 23,
  bandwidth: 156.4,
};

const initialTimeCrystal: TimeCrystal = {
  phase: 0.73,
  frequency: 1.618,
  coherence: 0.94,
  entropy: 0.12,
  synchronizedNodes: 847,
};

const initialCredits: CreditBalance = {
  available: 2847.50,
  pending: 156.25,
  earned: 4521.75,
  spent: 1674.25,
};

export const useNetworkStore = create<NetworkState>((set, get) => ({
  stats: initialStats,
  nodes: [],
  timeCrystal: initialTimeCrystal,
  credits: initialCredits,
  isConnected: true,
  isLoading: false,
  error: null,

  setStats: (stats) =>
    set((state) => ({ stats: { ...state.stats, ...stats } })),

  addNode: (node) =>
    set((state) => ({ nodes: [...state.nodes, node] })),

  removeNode: (nodeId) =>
    set((state) => ({
      nodes: state.nodes.filter((n) => n.id !== nodeId),
    })),

  updateNode: (nodeId, updates) =>
    set((state) => ({
      nodes: state.nodes.map((n) =>
        n.id === nodeId ? { ...n, ...updates } : n
      ),
    })),

  setTimeCrystal: (crystal) =>
    set((state) => ({
      timeCrystal: { ...state.timeCrystal, ...crystal },
    })),

  setCredits: (credits) =>
    set((state) => ({
      credits: { ...state.credits, ...credits },
    })),

  setConnected: (connected) => set({ isConnected: connected }),
  setLoading: (loading) => set({ isLoading: loading }),
  setError: (error) => set({ error }),

  simulateActivity: () => {
    const state = get();

    // Simulate network activity
    set({
      stats: {
        ...state.stats,
        activeNodes: state.stats.activeNodes + Math.floor(Math.random() * 10 - 5),
        totalCompute: state.stats.totalCompute + (Math.random() * 2 - 1),
        tasksCompleted: state.stats.tasksCompleted + Math.floor(Math.random() * 10),
        latency: Math.max(5, state.stats.latency + (Math.random() * 4 - 2)),
      },
      timeCrystal: {
        ...state.timeCrystal,
        phase: (state.timeCrystal.phase + 0.01) % 1,
        coherence: Math.min(1, Math.max(0.8, state.timeCrystal.coherence + (Math.random() * 0.02 - 0.01))),
      },
      credits: {
        ...state.credits,
        pending: state.credits.pending + Math.random() * 0.5,
      },
    });
  },
}));
