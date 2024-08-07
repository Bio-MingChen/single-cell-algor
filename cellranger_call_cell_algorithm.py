import numpy as np
import scanpy as sc

def estimate_recovered_cells_ordmag(nonzero_bc_counts, max_expected_cells):
    recovered_cells = np.linspace(1, np.log2(max_expected_cells), 2000)
    recovered_cells = np.unique(np.round(np.power(2, recovered_cells)).astype(int))
    baseline_bc_idx = np.round(recovered_cells * 0.01)
    baseline_bc_idx = np.minimum(baseline_bc_idx.astype(int), len(nonzero_bc_counts) - 1)
    filtered_cells = find_within_ordmag(nonzero_bc_counts, baseline_bc_idx)
    loss = np.power(filtered_cells - recovered_cells, 2) / recovered_cells
    idx = np.argmin(loss)
    return recovered_cells[idx], loss[idx]

def find_within_ordmag(x, baseline_idx):
    x_ascending = np.sort(x)
    baseline = x_ascending[-(baseline_idx + 1)]
    cutoff = np.maximum(1, np.round(0.1 * baseline)).astype(int)
    return len(x) - np.searchsorted(x_ascending, cutoff)


def call_cells(adata, max_expected_cells=None,num_bootstrap_samples=100,random_state=0):
    """
    Estimate the number of recovered cells and filter barcodes within an order of magnitude.
    """
    rs = np.random.RandomState(random_state)

    # 使用 `total_counts` 作为计数值
    nonzero_bc_counts = adata.obs['total_counts'].values
    nonzero_bc_counts = nonzero_bc_counts[nonzero_bc_counts > 0]

    if len(nonzero_bc_counts) == 0:
        print("WARNING: All barcodes do not have enough reads for ordmag, allowing no bcs through")
        return [], "WARNING: All barcodes do not have enough reads for ordmag, allowing no bcs through"

    if max_expected_cells is None:
        # max_expected_cells = 262144  # Default value if not provided
        max_expected_cells = 45000  # Default value if not provided

    bootstrap_samples = [
        estimate_recovered_cells_ordmag(
            rs.choice(nonzero_bc_counts, len(nonzero_bc_counts)), max_expected_cells
        )
        for _ in range(num_bootstrap_samples)
    ]

    bootstrap_samples_array = np.array(bootstrap_samples)
    recovered_cells, loss = np.mean(bootstrap_samples_array, axis=0)
    recovered_cells = max(int(np.round(recovered_cells)), 10)  # Minimum threshold for recovered cells

    print(f"Estimated recovered_cells = {recovered_cells} with average loss = {loss}")

    baseline_bc_idx = int(np.round(float(recovered_cells) * 0.01))
    baseline_bc_idx = min(baseline_bc_idx, len(nonzero_bc_counts) - 1)

    top_n_boot = np.array(
        [
            find_within_ordmag(
                rs.choice(nonzero_bc_counts, len(nonzero_bc_counts)), baseline_bc_idx
            )
            for _ in range(num_bootstrap_samples)
        ]
    )

    top_n = int(np.mean(top_n_boot))

    filtered_barcodes_idx = np.argsort(nonzero_bc_counts)[-top_n:]

    return filtered_barcodes_idx, f"Estimated recovered cells: {recovered_cells}, average loss: {loss}"
