"""
Neighbor Search Data Structures for SPH Simulations

This module implements efficient neighbor search using spatial hashing.

Key implementations:
- CellLinkedList: Divides domain into uniform grid cells for O(n) neighbor search
- Brute force search: For validation and small systems

The CellLinkedList approach:
1. Divide spatial domain into cubic/square cells of size cell_size
2. Assign each particle to its cell using spatial hash
3. Query cell and neighboring cells for close particles
"""

import numpy as np
from typing import List, Tuple, Dict, Optional


class CellLinkedList:
    """
    Cell-linked list for spatial neighbor search.

    Uses a dictionary-based spatial hash to divide the domain into uniform cells.
    Supports both 2D and 3D.

    Attributes:
        dim: Dimension (2 or 3)
        domain_min: Minimum corner of domain
        domain_max: Maximum corner of domain
        cell_size: Size of each grid cell
        cells: Dictionary mapping cell indices to particle indices
    """

    def __init__(
        self,
        domain_min: np.ndarray,
        domain_max: np.ndarray,
        cell_size: float,
    ):
        """
        Initialize the cell-linked list.

        Args:
            domain_min: Minimum coordinates of domain (shape: (2,) or (3,))
            domain_max: Maximum coordinates of domain (shape: (2,) or (3,))
            cell_size: Size of each cell (should be >= kernel support radius)
        """
        self.domain_min = np.asarray(domain_min, dtype=float)
        self.domain_max = np.asarray(domain_max, dtype=float)
        self.cell_size = float(cell_size)
        self.dim = len(self.domain_min)

        if self.dim not in (2, 3):
            raise ValueError("Only 2D and 3D are supported")

        # Ensure domain is valid
        if np.any(self.domain_max <= self.domain_min):
            raise ValueError("domain_max must be > domain_min")

        self.cells: Dict[Tuple, List[int]] = {}

    def _get_cell_index(self, position: np.ndarray) -> Tuple:
        """
        Get the cell index for a position.

        Args:
            position: Particle position (shape: (dim,))

        Returns:
            Cell index tuple
        """
        # Compute cell index
        cell_idx = np.floor((position - self.domain_min) / self.cell_size).astype(int)

        # Clamp to domain
        cell_idx = np.clip(
            cell_idx,
            0,
            np.floor((self.domain_max - self.domain_min) / self.cell_size).astype(int),
        )

        return tuple(cell_idx)

    def build(self, positions: np.ndarray) -> None:
        """
        Build the cell-linked list from particle positions.

        Args:
            positions: Particle positions (shape: (n_particles, dim))
        """
        positions = np.asarray(positions, dtype=float)

        if positions.ndim != 2:
            raise ValueError("positions must be 2D array")

        if positions.shape[1] != self.dim:
            raise ValueError(f"positions must have {self.dim} columns")

        # Clear existing cells
        self.cells.clear()

        # Assign particles to cells
        for particle_idx, position in enumerate(positions):
            cell_idx = self._get_cell_index(position)
            if cell_idx not in self.cells:
                self.cells[cell_idx] = []
            self.cells[cell_idx].append(particle_idx)

    def find_neighbors(
        self,
        particle_index: int,
        positions: np.ndarray,
        radius: float,
    ) -> np.ndarray:
        """
        Find neighbors of a particle within a given radius.

        Args:
            particle_index: Index of query particle
            positions: All particle positions (shape: (n_particles, dim))
            radius: Search radius

        Returns:
            Array of neighbor particle indices (excluding self)
        """
        positions = np.asarray(positions, dtype=float)
        radius = float(radius)

        if particle_index >= len(positions):
            raise ValueError(f"particle_index {particle_index} out of range")

        pos = positions[particle_index]
        neighbors = []

        # Get cell index of particle
        cell_idx = self._get_cell_index(pos)

        # Compute neighbor cells to check
        n_cells = int(np.ceil(radius / self.cell_size)) + 1
        offsets = self._get_neighbor_offsets(n_cells)

        # Check all neighbor cells
        for offset in offsets:
            neighbor_cell = tuple(np.array(cell_idx) + offset)
            if neighbor_cell not in self.cells:
                continue

            for neighbor_idx in self.cells[neighbor_cell]:
                if neighbor_idx == particle_index:
                    continue

                # Check distance
                dist = np.linalg.norm(positions[neighbor_idx] - pos)
                if dist <= radius:
                    neighbors.append(neighbor_idx)

        return np.array(neighbors, dtype=int)

    def find_all_neighbors(
        self,
        positions: np.ndarray,
        radius: float,
    ) -> List[np.ndarray]:
        """
        Find neighbors for all particles.

        Args:
            positions: All particle positions (shape: (n_particles, dim))
            radius: Search radius

        Returns:
            List of neighbor arrays, one per particle
        """
        positions = np.asarray(positions, dtype=float)
        n_particles = len(positions)

        neighbors_list = []
        for i in range(n_particles):
            neighbors = self.find_neighbors(i, positions, radius)
            neighbors_list.append(neighbors)

        return neighbors_list

    def _get_neighbor_offsets(self, n_cells: int) -> List[Tuple]:
        """
        Get all cell offsets for neighbor cells.

        Args:
            n_cells: Number of cells in each direction

        Returns:
            List of offset tuples
        """
        offsets = []

        if self.dim == 2:
            for dx in range(-n_cells, n_cells + 1):
                for dy in range(-n_cells, n_cells + 1):
                    offsets.append((dx, dy))
        else:  # dim == 3
            for dx in range(-n_cells, n_cells + 1):
                for dy in range(-n_cells, n_cells + 1):
                    for dz in range(-n_cells, n_cells + 1):
                        offsets.append((dx, dy, dz))

        return offsets

    def get_cell_statistics(self) -> Dict:
        """
        Get statistics about cell occupancy.

        Returns:
            Dictionary with statistics
        """
        n_cells = len(self.cells)
        n_particles = sum(len(particles) for particles in self.cells.values())

        if n_cells == 0:
            return {
                'n_cells': 0,
                'n_particles': 0,
                'avg_particles_per_cell': 0.0,
                'max_particles_in_cell': 0,
                'min_particles_in_cell': 0,
            }

        cell_counts = [len(particles) for particles in self.cells.values()]
        return {
            'n_cells': n_cells,
            'n_particles': n_particles,
            'avg_particles_per_cell': n_particles / n_cells,
            'max_particles_in_cell': max(cell_counts),
            'min_particles_in_cell': min(cell_counts),
        }


def find_neighbors_brute(
    particle_index: int,
    positions: np.ndarray,
    radius: float,
) -> np.ndarray:
    """
    Brute-force neighbor search (for validation/testing).

    Args:
        particle_index: Index of query particle
        positions: All particle positions (shape: (n_particles, dim))
        radius: Search radius

    Returns:
        Array of neighbor particle indices (excluding self)
    """
    positions = np.asarray(positions, dtype=float)
    radius = float(radius)

    if particle_index >= len(positions):
        raise ValueError(f"particle_index {particle_index} out of range")

    pos = positions[particle_index]

    # Compute distances to all particles
    distances = np.linalg.norm(positions - pos, axis=1)

    # Find neighbors within radius (excluding self)
    neighbors = np.where((distances <= radius) & (distances > 1e-10))[0]

    return neighbors


def find_all_neighbors_brute(
    positions: np.ndarray,
    radius: float,
) -> List[np.ndarray]:
    """
    Brute-force neighbor search for all particles (for validation/testing).

    Args:
        positions: All particle positions (shape: (n_particles, dim))
        radius: Search radius

    Returns:
        List of neighbor arrays, one per particle
    """
    positions = np.asarray(positions, dtype=float)
    n_particles = len(positions)

    neighbors_list = []
    for i in range(n_particles):
        neighbors = find_neighbors_brute(i, positions, radius)
        neighbors_list.append(neighbors)

    return neighbors_list


if __name__ == "__main__":
    """Test neighbor search implementations"""

    # Test 2D
    print("Testing 2D neighbor search...")
    positions_2d = np.array([
        [0.0, 0.0],
        [0.1, 0.0],
        [0.0, 0.1],
        [0.5, 0.5],
        [1.0, 1.0],
    ], dtype=float)

    # Cell-linked list
    domain_min = np.array([0.0, 0.0])
    domain_max = np.array([2.0, 2.0])
    cell_size = 0.2

    cll_2d = CellLinkedList(domain_min, domain_max, cell_size)
    cll_2d.build(positions_2d)

    # Find neighbors of particle 0 with radius 0.15
    neighbors_cll = cll_2d.find_neighbors(0, positions_2d, 0.15)
    neighbors_brute = find_neighbors_brute(0, positions_2d, 0.15)

    print(f"Neighbors (cell-linked list): {neighbors_cll}")
    print(f"Neighbors (brute force): {neighbors_brute}")
    print(f"Match: {set(neighbors_cll) == set(neighbors_brute)}")

    # Test 3D
    print("\nTesting 3D neighbor search...")
    positions_3d = np.random.rand(20, 3) * 2.0

    domain_min_3d = np.array([0.0, 0.0, 0.0])
    domain_max_3d = np.array([2.0, 2.0, 2.0])

    cll_3d = CellLinkedList(domain_min_3d, domain_max_3d, 0.3)
    cll_3d.build(positions_3d)

    neighbors_cll = cll_3d.find_neighbors(0, positions_3d, 0.4)
    neighbors_brute = find_neighbors_brute(0, positions_3d, 0.4)

    print(f"3D neighbors match: {set(neighbors_cll) == set(neighbors_brute)}")

    # Test find_all_neighbors
    all_neighbors_cll = cll_3d.find_all_neighbors(positions_3d, 0.4)
    all_neighbors_brute = find_all_neighbors_brute(positions_3d, 0.4)

    matches = sum(
        set(cll) == set(brute)
        for cll, brute in zip(all_neighbors_cll, all_neighbors_brute)
    )
    print(f"All neighbors match: {matches}/{len(positions_3d)}")

    # Statistics
    stats = cll_3d.get_cell_statistics()
    print(f"Cell statistics: {stats}")
