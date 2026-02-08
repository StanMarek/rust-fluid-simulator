use common::Dimension;

/// Struct of Arrays (SoA) particle storage for cache-friendly access
/// and easy GPU buffer mapping.
#[derive(Debug, Clone)]
pub struct ParticleStorage<D: Dimension> {
    pub positions: Vec<D::Vector>,
    pub velocities: Vec<D::Vector>,
    pub accelerations: Vec<D::Vector>,
    pub densities: Vec<f32>,
    pub pressures: Vec<f32>,
    pub masses: Vec<f32>,
}

impl<D: Dimension> ParticleStorage<D> {
    /// Create empty storage.
    pub fn new() -> Self {
        Self {
            positions: Vec::new(),
            velocities: Vec::new(),
            accelerations: Vec::new(),
            densities: Vec::new(),
            pressures: Vec::new(),
            masses: Vec::new(),
        }
    }

    /// Create storage with pre-allocated capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            positions: Vec::with_capacity(capacity),
            velocities: Vec::with_capacity(capacity),
            accelerations: Vec::with_capacity(capacity),
            densities: Vec::with_capacity(capacity),
            pressures: Vec::with_capacity(capacity),
            masses: Vec::with_capacity(capacity),
        }
    }

    /// Number of particles.
    pub fn len(&self) -> usize {
        self.positions.len()
    }

    /// Whether storage is empty.
    pub fn is_empty(&self) -> bool {
        self.positions.is_empty()
    }

    /// Add a single particle.
    pub fn add(&mut self, position: D::Vector, velocity: D::Vector, mass: f32) {
        self.positions.push(position);
        self.velocities.push(velocity);
        self.accelerations.push(D::zero());
        self.densities.push(0.0);
        self.pressures.push(0.0);
        self.masses.push(mass);
    }

    /// Remove a particle by index (swap-remove for O(1)).
    pub fn remove(&mut self, index: usize) {
        self.positions.swap_remove(index);
        self.velocities.swap_remove(index);
        self.accelerations.swap_remove(index);
        self.densities.swap_remove(index);
        self.pressures.swap_remove(index);
        self.masses.swap_remove(index);
    }

    /// Clear all particles.
    pub fn clear(&mut self) {
        self.positions.clear();
        self.velocities.clear();
        self.accelerations.clear();
        self.densities.clear();
        self.pressures.clear();
        self.masses.clear();
    }

    /// Zero out all accelerations (called at start of each step).
    pub fn clear_accelerations(&mut self) {
        for acc in &mut self.accelerations {
            *acc = D::zero();
        }
    }
}

impl<D: Dimension> Default for ParticleStorage<D> {
    fn default() -> Self {
        Self::new()
    }
}
