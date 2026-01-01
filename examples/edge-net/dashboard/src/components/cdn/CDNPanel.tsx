import { Button, Card, CardBody, Switch, Chip, Progress } from '@heroui/react';
import { motion } from 'framer-motion';
import { Download, Check, Package, Cpu, Shield, Network, Wrench } from 'lucide-react';
import { useCDNStore } from '../../stores/cdnStore';
import type { CDNScript } from '../../types';

const categoryIcons = {
  wasm: <Cpu size={16} />,
  ai: <Package size={16} />,
  crypto: <Shield size={16} />,
  network: <Network size={16} />,
  utility: <Wrench size={16} />,
};

const categoryColors = {
  wasm: 'bg-sky-500/20 text-sky-400 border-sky-500/30',
  ai: 'bg-violet-500/20 text-violet-400 border-violet-500/30',
  crypto: 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30',
  network: 'bg-cyan-500/20 text-cyan-400 border-cyan-500/30',
  utility: 'bg-amber-500/20 text-amber-400 border-amber-500/30',
};

export function CDNPanel() {
  const { scripts, autoLoad, cacheEnabled, isLoading, loadScript, unloadScript, toggleScript, setAutoLoad, setCacheEnabled } = useCDNStore();

  const groupedScripts = scripts.reduce((acc, script) => {
    if (!acc[script.category]) acc[script.category] = [];
    acc[script.category].push(script);
    return acc;
  }, {} as Record<string, CDNScript[]>);

  const loadedCount = scripts.filter((s) => s.loaded).length;

  return (
    <div className="space-y-6">
      {/* Header Stats */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <motion.div
          className="crystal-card p-4"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
        >
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-zinc-400">Scripts Loaded</p>
              <p className="text-2xl font-bold text-sky-400">{loadedCount}/{scripts.length}</p>
            </div>
            <div className="w-12 h-12 rounded-full bg-sky-500/20 flex items-center justify-center">
              <Download className="text-sky-400" size={24} />
            </div>
          </div>
          <Progress
            value={(loadedCount / scripts.length) * 100}
            className="mt-3"
            classNames={{
              indicator: 'bg-gradient-to-r from-sky-500 to-cyan-500',
              track: 'bg-zinc-800',
            }}
          />
        </motion.div>

        <motion.div
          className="crystal-card p-4"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
        >
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm text-zinc-400">Auto-Load</span>
            <Switch
              isSelected={autoLoad}
              onValueChange={setAutoLoad}
              size="sm"
              classNames={{
                wrapper: 'bg-zinc-700 group-data-[selected=true]:bg-sky-500',
              }}
            />
          </div>
          <p className="text-xs text-zinc-500">
            Automatically load enabled scripts on startup
          </p>
        </motion.div>

        <motion.div
          className="crystal-card p-4"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
        >
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm text-zinc-400">Cache Enabled</span>
            <Switch
              isSelected={cacheEnabled}
              onValueChange={setCacheEnabled}
              size="sm"
              classNames={{
                wrapper: 'bg-zinc-700 group-data-[selected=true]:bg-emerald-500',
              }}
            />
          </div>
          <p className="text-xs text-zinc-500">
            Cache loaded scripts in browser storage
          </p>
        </motion.div>
      </div>

      {/* Scripts by Category */}
      {Object.entries(groupedScripts).map(([category, categoryScripts], idx) => (
        <motion.div
          key={category}
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 * (idx + 3) }}
        >
          <div className="flex items-center gap-2 mb-3">
            <div className={`p-1.5 rounded ${categoryColors[category as keyof typeof categoryColors]}`}>
              {categoryIcons[category as keyof typeof categoryIcons]}
            </div>
            <h3 className="text-lg font-semibold capitalize">{category}</h3>
            <Chip size="sm" variant="flat" className="bg-zinc-800 text-zinc-400">
              {categoryScripts.length}
            </Chip>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            {categoryScripts.map((script) => (
              <Card
                key={script.id}
                className={`bg-zinc-900/50 border ${
                  script.loaded
                    ? 'border-emerald-500/30'
                    : script.enabled
                    ? 'border-sky-500/30'
                    : 'border-white/10'
                }`}
              >
                <CardBody className="p-4">
                  <div className="flex items-start justify-between gap-3">
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2">
                        <h4 className="font-medium text-white truncate">{script.name}</h4>
                        {script.loaded && (
                          <Check size={14} className="text-emerald-400 flex-shrink-0" />
                        )}
                      </div>
                      <p className="text-xs text-zinc-500 mt-1 line-clamp-2">
                        {script.description}
                      </p>
                      <div className="flex items-center gap-2 mt-2">
                        <Chip size="sm" variant="flat" className="bg-zinc-800 text-zinc-400">
                          {script.size}
                        </Chip>
                      </div>
                    </div>

                    <div className="flex flex-col gap-2">
                      <Switch
                        isSelected={script.enabled}
                        onValueChange={() => toggleScript(script.id)}
                        size="sm"
                        classNames={{
                          wrapper: 'bg-zinc-700 group-data-[selected=true]:bg-sky-500',
                        }}
                      />
                      <Button
                        size="sm"
                        variant="flat"
                        isDisabled={!script.enabled || isLoading}
                        isLoading={isLoading}
                        className={
                          script.loaded
                            ? 'bg-red-500/20 text-red-400'
                            : 'bg-sky-500/20 text-sky-400'
                        }
                        onPress={() =>
                          script.loaded ? unloadScript(script.id) : loadScript(script.id)
                        }
                      >
                        {script.loaded ? 'Unload' : 'Load'}
                      </Button>
                    </div>
                  </div>
                </CardBody>
              </Card>
            ))}
          </div>
        </motion.div>
      ))}
    </div>
  );
}
