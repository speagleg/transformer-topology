"""Analyze topological health of a reasoning CellComplex."""

import torch
from src.cell_complex.cell_complex import CellComplex
from src.spectral.decomposition import hodge_decomposition
from src.spectral.laplacian import hodge_laplacian_0
from src.metacog.health_report import ReasoningHealthReport


class TopologyAnalyzer:
    """Computes topological health metrics from a CellComplex snapshot."""

    def analyze(self, cc: CellComplex) -> ReasoningHealthReport:
        """Analyze the topological health of a reasoning graph.

        Args:
            cc: A CellComplex snapshot (should be cloned before passing).

        Returns:
            ReasoningHealthReport with all computed metrics.
        """
        num_nodes = cc.num_cells(0)
        num_edges = cc.num_cells(1)

        # Edge case: too few cells for meaningful analysis
        if num_nodes < 2 or num_edges == 0:
            return ReasoningHealthReport(
                curl_energy=0.0,
                gradient_energy=0.0,
                harmonic_energy=0.0,
                spectral_gap=0.0,
                connected_components=num_nodes,
                betti_1=0,
                density=0.0,
                num_nodes=num_nodes,
                num_edges=num_edges,
            )

        # Hodge decomposition on edge signal (dim 0 of edge embeddings = cosine sim)
        edge_embs = cc.get_embeddings(1)  # (num_edges, embedding_dim)
        edge_signal = edge_embs[:, 0]     # cosine similarity signal

        gradient, curl, harmonic = hodge_decomposition(cc, edge_signal, dim=1)

        # Energy ratios
        total_energy = (edge_signal ** 2).sum().item()
        if total_energy < 1e-12:
            curl_energy = 0.0
            gradient_energy = 0.0
            harmonic_energy = 0.0
        else:
            curl_energy = (curl ** 2).sum().item() / total_energy
            gradient_energy = (gradient ** 2).sum().item() / total_energy
            harmonic_energy = (harmonic ** 2).sum().item() / total_energy

        # Spectral gap (Fiedler value) from L0
        L0 = hodge_laplacian_0(cc).float()
        eigenvalues = torch.linalg.eigvalsh(L0)
        # Sort ascending (should already be, but ensure)
        eigenvalues = eigenvalues.sort().values

        # Count connected components = number of near-zero eigenvalues
        zero_threshold = 1e-5
        connected_components = (eigenvalues < zero_threshold).sum().item()
        connected_components = max(connected_components, 1)

        # Spectral gap = smallest non-zero eigenvalue
        nonzero_eigs = eigenvalues[eigenvalues >= zero_threshold]
        spectral_gap = nonzero_eigs[0].item() if len(nonzero_eigs) > 0 else 0.0

        # Betti-1 = edges - nodes + components (graph-theoretic)
        betti_1 = num_edges - num_nodes + connected_components

        # Graph density
        max_edges = num_nodes * (num_nodes - 1) / 2
        density = num_edges / max_edges if max_edges > 0 else 0.0

        return ReasoningHealthReport(
            curl_energy=curl_energy,
            gradient_energy=gradient_energy,
            harmonic_energy=harmonic_energy,
            spectral_gap=spectral_gap,
            connected_components=connected_components,
            betti_1=betti_1,
            density=density,
            num_nodes=num_nodes,
            num_edges=num_edges,
        )
