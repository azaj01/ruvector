import { Button, Navbar, NavbarBrand, NavbarContent, NavbarItem, Chip } from '@heroui/react';
import { motion } from 'framer-motion';
import { Activity, Wifi, WifiOff, Sun, Menu } from 'lucide-react';
import { useNetworkStore } from '../../stores/networkStore';

interface HeaderProps {
  onMenuToggle?: () => void;
  isMobile?: boolean;
}

export function Header({ onMenuToggle, isMobile }: HeaderProps) {
  const { isConnected, stats } = useNetworkStore();

  return (
    <Navbar
      className="bg-zinc-900/50 backdrop-blur-xl border-b border-white/10"
      maxWidth="full"
      height="4rem"
    >
      <NavbarContent>
        {isMobile && onMenuToggle && (
          <NavbarItem>
            <Button
              isIconOnly
              variant="light"
              onPress={onMenuToggle}
              className="text-zinc-400 hover:text-white"
            >
              <Menu size={20} />
            </Button>
          </NavbarItem>
        )}

        <NavbarBrand className="gap-3">
          {/* Crystal Logo */}
          <motion.div
            className="relative w-10 h-10"
            animate={{ rotate: 360 }}
            transition={{ duration: 20, repeat: Infinity, ease: 'linear' }}
          >
            <div
              className="absolute inset-0"
              style={{
                background: 'linear-gradient(135deg, #0ea5e9, #7c3aed, #06b6d4)',
                clipPath: 'polygon(50% 0%, 100% 50%, 50% 100%, 0% 50%)',
              }}
            />
            <motion.div
              className="absolute inset-2"
              style={{
                background: 'linear-gradient(135deg, #06b6d4, #0ea5e9)',
                clipPath: 'polygon(50% 0%, 100% 50%, 50% 100%, 0% 50%)',
              }}
              animate={{ opacity: [0.5, 1, 0.5] }}
              transition={{ duration: 2, repeat: Infinity }}
            />
          </motion.div>

          <div className="flex flex-col">
            <span className="font-bold text-lg bg-gradient-to-r from-sky-400 via-violet-400 to-cyan-400 bg-clip-text text-transparent">
              Edge-Net
            </span>
            <span className="text-[10px] text-zinc-500 -mt-1">Time Crystal Network</span>
          </div>
        </NavbarBrand>
      </NavbarContent>

      <NavbarContent justify="center" className="hidden md:flex gap-4">
        <NavbarItem>
          <Chip
            startContent={<Activity size={14} />}
            variant="flat"
            classNames={{
              base: 'bg-sky-500/10 border border-sky-500/30',
              content: 'text-sky-400 text-xs',
            }}
          >
            {stats.totalCompute.toFixed(1)} TFLOPS
          </Chip>
        </NavbarItem>

        <NavbarItem>
          <Chip
            startContent={
              <motion.div
                animate={{ scale: [1, 1.2, 1] }}
                transition={{ duration: 1, repeat: Infinity }}
              >
                <div className="w-2 h-2 rounded-full bg-emerald-400" />
              </motion.div>
            }
            variant="flat"
            classNames={{
              base: 'bg-emerald-500/10 border border-emerald-500/30',
              content: 'text-emerald-400 text-xs',
            }}
          >
            {stats.activeNodes.toLocaleString()} nodes
          </Chip>
        </NavbarItem>
      </NavbarContent>

      <NavbarContent justify="end" className="gap-2">
        <NavbarItem>
          <motion.div
            animate={isConnected ? { opacity: [0.5, 1, 0.5] } : {}}
            transition={{ duration: 2, repeat: Infinity }}
          >
            <Chip
              startContent={isConnected ? <Wifi size={14} /> : <WifiOff size={14} />}
              variant="flat"
              classNames={{
                base: isConnected
                  ? 'bg-emerald-500/10 border border-emerald-500/30'
                  : 'bg-red-500/10 border border-red-500/30',
                content: isConnected ? 'text-emerald-400' : 'text-red-400',
              }}
            >
              {isConnected ? 'Connected' : 'Offline'}
            </Chip>
          </motion.div>
        </NavbarItem>

        <NavbarItem className="hidden sm:flex">
          <Button isIconOnly variant="light" className="text-zinc-400 hover:text-white">
            <Sun size={18} />
          </Button>
        </NavbarItem>
      </NavbarContent>
    </Navbar>
  );
}
