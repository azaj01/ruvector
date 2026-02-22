use serde::{Deserialize, Serialize};

/// Budget allocation for each processing phase within a container epoch.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContainerEpochBudget {
    /// Total tick budget for the entire epoch.
    pub total: u64,
    /// Ticks allocated to the ingest phase.
    pub ingest: u64,
    /// Ticks allocated to the min-cut phase.
    pub mincut: u64,
    /// Ticks allocated to the spectral analysis phase.
    pub spectral: u64,
    /// Ticks allocated to the evidence accumulation phase.
    pub evidence: u64,
    /// Ticks allocated to the witness generation phase.
    pub witness: u64,
}

impl Default for ContainerEpochBudget {
    fn default() -> Self {
        Self {
            total: 10_000,
            ingest: 2_000,
            mincut: 3_000,
            spectral: 2_000,
            evidence: 2_000,
            witness: 1_000,
        }
    }
}

/// Processing phases within a container epoch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Phase {
    Ingest,
    MinCut,
    Spectral,
    Evidence,
    Witness,
}

/// Controls epoch budgeting and tracks per-phase tick consumption.
pub struct EpochController {
    budget: ContainerEpochBudget,
    ticks_used: u64,
    phase_ticks: [u64; 5],
    current_phase: Phase,
}

impl EpochController {
    /// Creates a new epoch controller with the given budget.
    pub fn new(budget: ContainerEpochBudget) -> Self {
        Self {
            budget,
            ticks_used: 0,
            phase_ticks: [0; 5],
            current_phase: Phase::Ingest,
        }
    }

    /// Checks whether the given phase has remaining budget.
    ///
    /// If the phase has budget, sets it as the current phase and returns `true`.
    /// Returns `false` if the phase budget or total budget is exhausted.
    pub fn try_budget(&mut self, phase: Phase) -> bool {
        let phase_budget = self.phase_budget(phase);
        let phase_used = self.phase_used(phase);

        if phase_used >= phase_budget || self.ticks_used >= self.budget.total {
            return false;
        }

        self.current_phase = phase;
        true
    }

    /// Consumes the given number of ticks from the current phase.
    pub fn consume(&mut self, ticks: u64) {
        let idx = Self::phase_index(self.current_phase);
        self.phase_ticks[idx] += ticks;
        self.ticks_used += ticks;
    }

    /// Returns the remaining ticks in the total budget.
    pub fn remaining(&self) -> u64 {
        self.budget.total.saturating_sub(self.ticks_used)
    }

    /// Resets all tick counters, restoring the full budget.
    pub fn reset(&mut self) {
        self.ticks_used = 0;
        self.phase_ticks = [0; 5];
        self.current_phase = Phase::Ingest;
    }

    /// Returns the budget allocated to the given phase.
    pub fn phase_budget(&self, phase: Phase) -> u64 {
        match phase {
            Phase::Ingest => self.budget.ingest,
            Phase::MinCut => self.budget.mincut,
            Phase::Spectral => self.budget.spectral,
            Phase::Evidence => self.budget.evidence,
            Phase::Witness => self.budget.witness,
        }
    }

    /// Returns the number of ticks consumed in the given phase.
    pub fn phase_used(&self, phase: Phase) -> u64 {
        self.phase_ticks[Self::phase_index(phase)]
    }

    /// Returns the total number of ticks consumed across all phases.
    pub fn total_used(&self) -> u64 {
        self.ticks_used
    }

    /// Returns the current active phase.
    pub fn current_phase(&self) -> Phase {
        self.current_phase
    }

    /// Returns the budget configuration.
    pub fn budget(&self) -> &ContainerEpochBudget {
        &self.budget
    }

    fn phase_index(phase: Phase) -> usize {
        match phase {
            Phase::Ingest => 0,
            Phase::MinCut => 1,
            Phase::Spectral => 2,
            Phase::Evidence => 3,
            Phase::Witness => 4,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_epoch_budgeting() {
        let budget = ContainerEpochBudget::default();
        let mut ctrl = EpochController::new(budget);

        // All phases should have budget initially
        assert!(ctrl.try_budget(Phase::Ingest));
        assert_eq!(ctrl.remaining(), 10_000);

        // Consume some ingest ticks
        ctrl.consume(1_500);
        assert_eq!(ctrl.phase_used(Phase::Ingest), 1_500);
        assert_eq!(ctrl.remaining(), 8_500);

        // Ingest still has budget (2000 - 1500 = 500 remaining)
        assert!(ctrl.try_budget(Phase::Ingest));

        // Exhaust ingest budget
        ctrl.consume(500);
        assert_eq!(ctrl.phase_used(Phase::Ingest), 2_000);
        // Ingest is now exhausted
        assert!(!ctrl.try_budget(Phase::Ingest));

        // MinCut should still have budget
        assert!(ctrl.try_budget(Phase::MinCut));
        ctrl.consume(3_000);
        assert!(!ctrl.try_budget(Phase::MinCut));

        // After reset, everything is fresh
        ctrl.reset();
        assert_eq!(ctrl.remaining(), 10_000);
        assert!(ctrl.try_budget(Phase::Ingest));
        assert_eq!(ctrl.phase_used(Phase::Ingest), 0);
    }

    #[test]
    fn test_epoch_total_budget_exhaustion() {
        let budget = ContainerEpochBudget {
            total: 100,
            ingest: 50,
            mincut: 50,
            spectral: 50,
            evidence: 50,
            witness: 50,
        };
        let mut ctrl = EpochController::new(budget);

        // Consume all of the total budget via ingest
        assert!(ctrl.try_budget(Phase::Ingest));
        ctrl.consume(50);
        assert!(ctrl.try_budget(Phase::MinCut));
        ctrl.consume(50);

        // Total budget is now exhausted -- no phase can run
        assert!(!ctrl.try_budget(Phase::Spectral));
        assert!(!ctrl.try_budget(Phase::Evidence));
        assert!(!ctrl.try_budget(Phase::Witness));
    }

    #[test]
    fn test_epoch_default_budget() {
        let budget = ContainerEpochBudget::default();
        assert_eq!(budget.total, 10_000);
        assert_eq!(
            budget.ingest + budget.mincut + budget.spectral + budget.evidence + budget.witness,
            10_000
        );
    }

    #[test]
    fn test_epoch_phase_index_roundtrip() {
        let phases = [
            Phase::Ingest,
            Phase::MinCut,
            Phase::Spectral,
            Phase::Evidence,
            Phase::Witness,
        ];
        for (i, phase) in phases.iter().enumerate() {
            assert_eq!(EpochController::phase_index(*phase), i);
        }
    }
}
