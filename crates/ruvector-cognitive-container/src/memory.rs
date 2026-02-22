use serde::{Deserialize, Serialize};

use crate::error::{ContainerError, Result};

/// Configuration for the memory slab layout.
///
/// Each budget represents the number of bytes allocated to that component
/// within the overall slab.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryConfig {
    /// Total slab size in bytes.
    pub slab_size: usize,
    /// Bytes reserved for graph adjacency data.
    pub graph_budget: usize,
    /// Bytes reserved for feature/embedding storage.
    pub feature_budget: usize,
    /// Bytes reserved for solver scratch space.
    pub solver_budget: usize,
    /// Bytes reserved for witness chain receipts.
    pub witness_budget: usize,
    /// Bytes reserved for evidence accumulation.
    pub evidence_budget: usize,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        // 4 MB total, divided among components
        Self {
            slab_size: 4 * 1024 * 1024,
            graph_budget: 1024 * 1024,       // 1 MB
            feature_budget: 1024 * 1024,     // 1 MB
            solver_budget: 512 * 1024,       // 512 KB
            witness_budget: 512 * 1024,      // 512 KB
            evidence_budget: 1024 * 1024,    // 1 MB
        }
    }
}

impl MemoryConfig {
    /// Validates that the component budgets do not exceed the slab size.
    pub fn validate(&self) -> Result<()> {
        let total = self.graph_budget
            + self.feature_budget
            + self.solver_budget
            + self.witness_budget
            + self.evidence_budget;

        if total > self.slab_size {
            return Err(ContainerError::InvalidConfig {
                reason: format!(
                    "component budgets ({total} bytes) exceed slab size ({} bytes)",
                    self.slab_size
                ),
            });
        }
        Ok(())
    }
}

/// A contiguous memory slab backing all container allocations.
pub struct MemorySlab {
    data: Vec<u8>,
    config: MemoryConfig,
}

impl MemorySlab {
    /// Creates a new memory slab with the given configuration.
    pub fn new(config: MemoryConfig) -> Result<Self> {
        config.validate()?;
        let data = vec![0u8; config.slab_size];
        Ok(Self { data, config })
    }

    /// Returns the total size of the slab in bytes.
    pub fn total_size(&self) -> usize {
        self.data.len()
    }

    /// Returns a read-only view of the entire slab.
    pub fn as_bytes(&self) -> &[u8] {
        &self.data
    }

    /// Returns the memory configuration.
    pub fn config(&self) -> &MemoryConfig {
        &self.config
    }
}

/// A bump-allocator arena within a fixed region of memory.
///
/// Supports fast sequential allocation and bulk reset. Individual
/// deallocations are not supported -- call `reset()` to reclaim all space.
pub struct Arena {
    base_offset: usize,
    size: usize,
    offset: usize,
}

impl Arena {
    /// Creates a new arena covering `[base_offset, base_offset + size)`.
    pub fn new(base_offset: usize, size: usize) -> Self {
        Self {
            base_offset,
            size,
            offset: 0,
        }
    }

    /// Allocates `size` bytes with the given alignment.
    ///
    /// Returns the absolute offset within the parent slab on success.
    pub fn alloc(&mut self, size: usize, align: usize) -> Result<usize> {
        let align = align.max(1);
        // Round current offset up to alignment boundary
        let aligned_offset = (self.offset + align - 1) & !(align - 1);
        let end = aligned_offset + size;

        if end > self.size {
            return Err(ContainerError::AllocationFailed {
                requested: size,
                available: self.size.saturating_sub(aligned_offset),
            });
        }

        self.offset = end;
        Ok(self.base_offset + aligned_offset)
    }

    /// Resets the arena, reclaiming all allocated space.
    pub fn reset(&mut self) {
        self.offset = 0;
    }

    /// Returns the number of bytes currently allocated.
    pub fn used(&self) -> usize {
        self.offset
    }

    /// Returns the number of bytes remaining in the arena.
    pub fn remaining(&self) -> usize {
        self.size.saturating_sub(self.offset)
    }

    /// Returns the base offset of this arena within the parent slab.
    pub fn base_offset(&self) -> usize {
        self.base_offset
    }

    /// Returns the total size of this arena.
    pub fn size(&self) -> usize {
        self.size
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_slab_creation() {
        let config = MemoryConfig::default();
        let slab = MemorySlab::new(config).unwrap();
        assert_eq!(slab.total_size(), 4 * 1024 * 1024);
        assert_eq!(slab.as_bytes().len(), 4 * 1024 * 1024);
        // Slab should be zero-initialized
        assert!(slab.as_bytes().iter().all(|&b| b == 0));
    }

    #[test]
    fn test_memory_config_validation_ok() {
        let config = MemoryConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_memory_config_validation_overflow() {
        let config = MemoryConfig {
            slab_size: 100,
            graph_budget: 50,
            feature_budget: 50,
            solver_budget: 50,
            witness_budget: 0,
            evidence_budget: 0,
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_arena_allocation() {
        let mut arena = Arena::new(0, 1024);
        assert_eq!(arena.used(), 0);
        assert_eq!(arena.remaining(), 1024);

        // Allocate 64 bytes aligned to 8
        let offset1 = arena.alloc(64, 8).unwrap();
        assert_eq!(offset1, 0);
        assert_eq!(arena.used(), 64);
        assert_eq!(arena.remaining(), 960);

        // Allocate another 128 bytes aligned to 16
        let offset2 = arena.alloc(128, 16).unwrap();
        assert_eq!(offset2, 64); // 64 is already 16-aligned
        assert_eq!(arena.used(), 192);

        // Reset reclaims all space
        arena.reset();
        assert_eq!(arena.used(), 0);
        assert_eq!(arena.remaining(), 1024);
    }

    #[test]
    fn test_arena_alignment() {
        let mut arena = Arena::new(0, 1024);

        // Allocate 3 bytes (non-aligned size)
        let offset1 = arena.alloc(3, 1).unwrap();
        assert_eq!(offset1, 0);
        assert_eq!(arena.used(), 3);

        // Next allocation with alignment 8 should pad to offset 8
        let offset2 = arena.alloc(10, 8).unwrap();
        assert_eq!(offset2, 8);
        assert_eq!(arena.used(), 18);
    }

    #[test]
    fn test_arena_overflow() {
        let mut arena = Arena::new(0, 64);
        // This should succeed
        arena.alloc(32, 1).unwrap();
        // This should fail -- only 32 remaining, requesting 64
        let result = arena.alloc(64, 1);
        assert!(result.is_err());
    }

    #[test]
    fn test_arena_base_offset() {
        let mut arena = Arena::new(4096, 256);
        let offset = arena.alloc(16, 8).unwrap();
        // Should be relative to base_offset
        assert_eq!(offset, 4096);

        let offset2 = arena.alloc(16, 8).unwrap();
        assert_eq!(offset2, 4096 + 16);
    }
}
