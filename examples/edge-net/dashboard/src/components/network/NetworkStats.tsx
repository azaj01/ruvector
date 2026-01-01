import { motion } from 'framer-motion';
import { Activity, Cpu, Users, Zap, Clock, Gauge } from 'lucide-react';
import { useNetworkStore } from '../../stores/networkStore';
import { StatCard } from '../common/StatCard';

export function NetworkStats() {
  const { stats, timeCrystal } = useNetworkStore();

  const statItems = [
    {
      title: 'Active Nodes',
      value: stats.activeNodes,
      icon: <Users size={24} />,
      color: 'crystal' as const,
      change: 2.4,
    },
    {
      title: 'Total Compute',
      value: `${stats.totalCompute.toFixed(1)} TFLOPS`,
      icon: <Cpu size={24} />,
      color: 'temporal' as const,
      change: 5.2,
    },
    {
      title: 'Tasks Completed',
      value: stats.tasksCompleted,
      icon: <Activity size={24} />,
      color: 'quantum' as const,
      change: 12.8,
    },
    {
      title: 'Credits Earned',
      value: `${stats.creditsEarned.toLocaleString()}`,
      icon: <Zap size={24} />,
      color: 'success' as const,
      change: 8.3,
    },
    {
      title: 'Network Latency',
      value: `${stats.latency.toFixed(0)}ms`,
      icon: <Clock size={24} />,
      color: stats.latency < 50 ? 'success' as const : 'warning' as const,
      change: stats.latency < 50 ? -3.2 : 1.5,
    },
    {
      title: 'Uptime',
      value: `${stats.uptime.toFixed(1)}%`,
      icon: <Gauge size={24} />,
      color: stats.uptime > 99 ? 'success' as const : 'warning' as const,
    },
  ];

  return (
    <div className="space-y-6">
      {/* Main Stats Grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
        {statItems.map((stat, index) => (
          <motion.div
            key={stat.title}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: index * 0.1 }}
          >
            <StatCard {...stat} />
          </motion.div>
        ))}
      </div>

      {/* Time Crystal Status */}
      <motion.div
        className="crystal-card p-6"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.6 }}
      >
        <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
          <motion.div
            className="w-3 h-3 rounded-full bg-gradient-to-r from-sky-400 to-violet-400"
            animate={{ scale: [1, 1.2, 1] }}
            transition={{ duration: 2, repeat: Infinity }}
          />
          Time Crystal Synchronization
        </h3>

        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="text-center p-4 rounded-lg bg-sky-500/10 border border-sky-500/20">
            <p className="text-2xl font-bold text-sky-400">
              {(timeCrystal.phase * 100).toFixed(0)}%
            </p>
            <p className="text-xs text-zinc-400 mt-1">Phase</p>
          </div>

          <div className="text-center p-4 rounded-lg bg-violet-500/10 border border-violet-500/20">
            <p className="text-2xl font-bold text-violet-400">
              {timeCrystal.frequency.toFixed(3)}
            </p>
            <p className="text-xs text-zinc-400 mt-1">Frequency (φ)</p>
          </div>

          <div className="text-center p-4 rounded-lg bg-cyan-500/10 border border-cyan-500/20">
            <p className="text-2xl font-bold text-cyan-400">
              {(timeCrystal.coherence * 100).toFixed(1)}%
            </p>
            <p className="text-xs text-zinc-400 mt-1">Coherence</p>
          </div>

          <div className="text-center p-4 rounded-lg bg-emerald-500/10 border border-emerald-500/20">
            <p className="text-2xl font-bold text-emerald-400">
              {timeCrystal.synchronizedNodes}
            </p>
            <p className="text-xs text-zinc-400 mt-1">Synced Nodes</p>
          </div>
        </div>

        {/* Crystal Animation */}
        <div className="mt-6 h-2 bg-zinc-800 rounded-full overflow-hidden">
          <motion.div
            className="h-full bg-gradient-to-r from-sky-500 via-violet-500 to-cyan-500"
            style={{ width: `${timeCrystal.coherence * 100}%` }}
            animate={{
              opacity: [0.7, 1, 0.7],
            }}
            transition={{ duration: 2, repeat: Infinity }}
          />
        </div>
      </motion.div>
    </div>
  );
}
