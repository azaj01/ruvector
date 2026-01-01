import { describe, it, expect, beforeEach } from 'vitest';
import { useNetworkStore } from '../stores/networkStore';
import { useWASMStore } from '../stores/wasmStore';
import { useMCPStore } from '../stores/mcpStore';
import { useCDNStore } from '../stores/cdnStore';

describe('Network Store', () => {
  beforeEach(() => {
    useNetworkStore.setState({
      stats: {
        totalNodes: 1247,
        activeNodes: 892,
        totalCompute: 45.8,
        creditsEarned: 12847,
        tasksCompleted: 89432,
        uptime: 99.7,
        latency: 23,
        bandwidth: 156.4,
      },
      isConnected: true,
      isLoading: false,
      error: null,
    });
  });

  it('should have initial stats', () => {
    const { stats } = useNetworkStore.getState();
    expect(stats.totalNodes).toBe(1247);
    expect(stats.activeNodes).toBe(892);
  });

  it('should update stats', () => {
    const { setStats } = useNetworkStore.getState();
    setStats({ activeNodes: 900 });

    const { stats } = useNetworkStore.getState();
    expect(stats.activeNodes).toBe(900);
  });

  it('should simulate activity', () => {
    useNetworkStore.getState().simulateActivity();

    const { stats } = useNetworkStore.getState();
    // Tasks should change (could be higher or lower due to randomness)
    expect(typeof stats.tasksCompleted).toBe('number');
  });

  it('should track connection status', () => {
    const { setConnected } = useNetworkStore.getState();

    setConnected(false);
    expect(useNetworkStore.getState().isConnected).toBe(false);

    setConnected(true);
    expect(useNetworkStore.getState().isConnected).toBe(true);
  });
});

describe('WASM Store', () => {
  it('should have default modules', () => {
    const { modules } = useWASMStore.getState();
    expect(modules.length).toBeGreaterThan(0);
    expect(modules[0].id).toBe('edge-net');
  });

  it('should update module status', () => {
    const { updateModule } = useWASMStore.getState();

    updateModule('edge-net', { status: 'loading' });

    const updatedModules = useWASMStore.getState().modules;
    const edgeNet = updatedModules.find((m) => m.id === 'edge-net');
    expect(edgeNet?.status).toBe('loading');
  });

  it('should track benchmarks', () => {
    const { addBenchmark, benchmarks } = useWASMStore.getState();
    const initialCount = benchmarks.length;

    addBenchmark({
      moduleId: 'edge-net',
      operation: 'test',
      iterations: 1000,
      avgTime: 0.5,
      minTime: 0.1,
      maxTime: 1.0,
      throughput: 2000,
    });

    expect(useWASMStore.getState().benchmarks.length).toBe(initialCount + 1);
  });
});

describe('MCP Store', () => {
  it('should have default tools', () => {
    const { tools } = useMCPStore.getState();
    expect(tools.length).toBeGreaterThan(0);
    expect(tools.some((t) => t.category === 'swarm')).toBe(true);
  });

  it('should update tool status', () => {
    const { updateTool } = useMCPStore.getState();

    updateTool('swarm_init', { status: 'running' });

    const updatedTools = useMCPStore.getState().tools;
    const tool = updatedTools.find((t) => t.id === 'swarm_init');
    expect(tool?.status).toBe('running');
  });

  it('should add results', () => {
    const { addResult } = useMCPStore.getState();

    addResult({
      toolId: 'swarm_init',
      success: true,
      data: { test: true },
      timestamp: new Date(),
      duration: 100,
    });

    const { results } = useMCPStore.getState();
    expect(results.length).toBeGreaterThan(0);
  });
});

describe('CDN Store', () => {
  it('should have default scripts', () => {
    const { scripts } = useCDNStore.getState();
    expect(scripts.length).toBeGreaterThan(0);
    expect(scripts.some((s) => s.category === 'wasm')).toBe(true);
  });

  it('should toggle script enabled state', () => {
    const { toggleScript, scripts } = useCDNStore.getState();
    const initialEnabled = scripts[0].enabled;

    toggleScript(scripts[0].id);

    const updatedScripts = useCDNStore.getState().scripts;
    expect(updatedScripts[0].enabled).toBe(!initialEnabled);
  });

  it('should track auto-load setting', () => {
    const { setAutoLoad } = useCDNStore.getState();

    setAutoLoad(true);
    expect(useCDNStore.getState().autoLoad).toBe(true);

    setAutoLoad(false);
    expect(useCDNStore.getState().autoLoad).toBe(false);
  });
});
