"""Tests for geometric invariance contrastive learning components."""

import numpy as np
import pytest
import torch
import xarray as xr

from lisbet.datasets import GeometricInvarianceDataset
from lisbet.modeling import AlignmentMetric, InfoNCELoss, ProjectionHead, UniformityMetric


@pytest.fixture
def sample_window():
    """Create a sample window of pose data for testing."""
    # Shape: (time=16, individuals=2, keypoints=4, space=2)
    data = np.random.randn(16, 2, 4, 2).astype(np.float32) * 0.1 + 0.5
    
    window = xr.DataArray(
        data,
        dims=["time", "individuals", "keypoints", "space"],
        coords={
            "time": np.arange(16),
            "individuals": np.arange(2),
            "keypoints": np.arange(4),
            "space": ["x", "y"],
        },
    )
    return window


@pytest.fixture
def sample_record(sample_window):
    """Create a sample record with pose data."""
    # Extend time dimension to simulate a longer sequence
    data = np.random.randn(100, 2, 4, 2).astype(np.float32) * 0.1 + 0.5
    
    record = xr.DataArray(
        data,
        dims=["time", "individuals", "keypoints", "space"],
        coords={
            "time": np.arange(100),
            "individuals": np.arange(2),
            "keypoints": np.arange(4),
            "space": ["x", "y"],
        },
    )
    
    # Wrap in object to simulate record structure
    class Record:
        def __init__(self, data):
            self.data = data
    
    return Record(record)


class TestProjectionHead:
    """Test ProjectionHead for contrastive learning."""

    def test_initialization(self):
        """Test ProjectionHead can be initialized with different parameters."""
        head = ProjectionHead(
            input_dim=256,
            hidden_dim=512,
            output_dim=128,
            use_batch_norm=True,
            normalize_output=True,
        )
        assert head is not None
        assert head.input_dim == 256
        assert head.hidden_dim == 512
        assert head.output_dim == 128

    def test_forward_pass(self):
        """Test ProjectionHead forward pass."""
        batch_size = 8
        input_dim = 256
        output_dim = 128
        
        head = ProjectionHead(
            input_dim=input_dim,
            hidden_dim=512,
            output_dim=output_dim,
            use_batch_norm=True,
            normalize_output=True,
        )
        
        # Create random input
        x = torch.randn(batch_size, input_dim)
        
        # Forward pass
        output = head(x)
        
        # Check output shape
        assert output.shape == (batch_size, output_dim)
        
        # Check L2 normalization (if enabled)
        norms = torch.norm(output, p=2, dim=1)
        assert torch.allclose(norms, torch.ones(batch_size), atol=1e-5)

    def test_no_batch_norm(self):
        """Test ProjectionHead without batch normalization."""
        head = ProjectionHead(
            input_dim=256,
            hidden_dim=512,
            output_dim=128,
            use_batch_norm=False,
            normalize_output=False,
        )
        
        x = torch.randn(8, 256)
        output = head(x)
        
        assert output.shape == (8, 128)


class TestInfoNCELoss:
    """Test InfoNCE loss for contrastive learning."""

    def test_initialization(self):
        """Test InfoNCELoss can be initialized."""
        loss_fn = InfoNCELoss(temperature=0.07)
        assert loss_fn is not None
        assert loss_fn.temperature == 0.07

    def test_forward_pass(self):
        """Test InfoNCELoss forward pass with perfect matches."""
        batch_size = 8
        embedding_dim = 128
        
        loss_fn = InfoNCELoss(temperature=0.07)
        
        # Create identical embeddings (perfect positive pairs)
        z1 = torch.randn(batch_size, embedding_dim)
        z1 = torch.nn.functional.normalize(z1, p=2, dim=1)
        z2 = z1.clone()
        
        # Loss should be close to 0 for perfect matches
        loss = loss_fn(z1, z2)
        
        assert loss.item() >= 0  # Loss is always non-negative
        assert torch.isfinite(loss)

    def test_different_embeddings(self):
        """Test InfoNCELoss with different embeddings."""
        batch_size = 8
        embedding_dim = 128
        
        loss_fn = InfoNCELoss(temperature=0.07)
        
        # Create different normalized embeddings
        z1 = torch.nn.functional.normalize(torch.randn(batch_size, embedding_dim), p=2, dim=1)
        z2 = torch.nn.functional.normalize(torch.randn(batch_size, embedding_dim), p=2, dim=1)
        
        # Loss should be positive for non-matching pairs
        loss = loss_fn(z1, z2)
        
        assert loss.item() > 0
        assert torch.isfinite(loss)

    def test_temperature_effect(self):
        """Test that temperature affects the loss value."""
        batch_size = 8
        embedding_dim = 128
        
        z1 = torch.nn.functional.normalize(torch.randn(batch_size, embedding_dim), p=2, dim=1)
        z2 = torch.nn.functional.normalize(torch.randn(batch_size, embedding_dim), p=2, dim=1)
        
        # Lower temperature should give higher loss
        loss_low_temp = InfoNCELoss(temperature=0.01)(z1, z2)
        loss_high_temp = InfoNCELoss(temperature=1.0)(z1, z2)
        
        assert loss_low_temp.item() > loss_high_temp.item()


class TestAlignmentMetric:
    """Test Alignment metric for contrastive learning."""

    def test_initialization(self):
        """Test AlignmentMetric can be initialized."""
        metric = AlignmentMetric()
        assert metric is not None

    def test_perfect_alignment(self):
        """Test metric with perfect alignment (identical embeddings)."""
        metric = AlignmentMetric()
        
        batch_size = 8
        embedding_dim = 128
        
        # Identical embeddings
        z1 = torch.randn(batch_size, embedding_dim)
        z2 = z1.clone()
        
        metric.update(z1, z2)
        alignment = metric.compute()
        
        # Perfect alignment should be close to 0
        assert alignment.item() < 0.1
        assert torch.isfinite(alignment)

    def test_random_alignment(self):
        """Test metric with random embeddings."""
        metric = AlignmentMetric()
        
        batch_size = 8
        embedding_dim = 128
        
        # Random embeddings
        z1 = torch.randn(batch_size, embedding_dim)
        z2 = torch.randn(batch_size, embedding_dim)
        
        metric.update(z1, z2)
        alignment = metric.compute()
        
        # Random alignment should be positive
        assert alignment.item() > 0
        assert torch.isfinite(alignment)


class TestUniformityMetric:
    """Test Uniformity metric for contrastive learning."""

    def test_initialization(self):
        """Test UniformityMetric can be initialized."""
        metric = UniformityMetric()
        assert metric is not None

    def test_uniform_distribution(self):
        """Test metric with uniformly distributed embeddings."""
        metric = UniformityMetric()
        
        batch_size = 100
        embedding_dim = 128
        
        # Normalized random embeddings (approximately uniform on hypersphere)
        z = torch.nn.functional.normalize(torch.randn(batch_size, embedding_dim), p=2, dim=1)
        
        metric.update(z)
        uniformity = metric.compute()
        
        # Uniformity should be negative (good distribution)
        assert uniformity.item() < 0
        assert torch.isfinite(uniformity)

    def test_collapsed_distribution(self):
        """Test metric with collapsed embeddings."""
        metric = UniformityMetric()
        
        batch_size = 100
        embedding_dim = 128
        
        # All embeddings are the same (collapsed)
        z = torch.ones(batch_size, embedding_dim)
        z = torch.nn.functional.normalize(z, p=2, dim=1)
        
        metric.update(z)
        uniformity = metric.compute()
        
        # Collapsed distribution should have bad uniformity
        assert torch.isfinite(uniformity)


class TestGeometricInvarianceDataset:
    """Test GeometricInvarianceDataset."""

    def test_initialization(self, sample_record):
        """Test dataset can be initialized."""
        dataset = GeometricInvarianceDataset(
            records=[sample_record],
            window_size=16,
            window_offset=0,
            fps_scaling=1.0,
            transform=None,
            base_seed=42,
        )
        assert dataset is not None

    def test_yields_pairs(self, sample_record):
        """Test dataset yields pairs of views."""
        dataset = GeometricInvarianceDataset(
            records=[sample_record],
            window_size=16,
            window_offset=0,
            fps_scaling=1.0,
            transform=None,
            base_seed=42,
        )
        
        # Get one sample
        iterator = iter(dataset)
        x_orig, x_transform = next(iterator)
        
        # Check both views are xarray DataArrays
        assert isinstance(x_orig, xr.DataArray)
        assert isinstance(x_transform, xr.DataArray)
        
        # Check shapes match
        assert x_orig.shape == x_transform.shape

    def test_geometric_transformation_applied(self, sample_record):
        """Test that geometric transformations are applied."""
        dataset = GeometricInvarianceDataset(
            records=[sample_record],
            window_size=16,
            window_offset=0,
            fps_scaling=1.0,
            transform=None,
            base_seed=42,
        )
        
        # Get one sample
        iterator = iter(dataset)
        x_orig, x_transform = next(iterator)
        
        # Check transformation attributes are present
        assert "rotation_angle" in x_transform.attrs
        assert "flip_horizontal" in x_transform.attrs
        assert "flip_vertical" in x_transform.attrs
        
        # Check that values differ (transformation was applied)
        assert not np.allclose(x_orig.values, x_transform.values)

    def test_reproducibility(self, sample_record):
        """Test that same seed produces same transformations."""
        seed = 42
        
        dataset1 = GeometricInvarianceDataset(
            records=[sample_record],
            window_size=16,
            window_offset=0,
            fps_scaling=1.0,
            transform=None,
            base_seed=seed,
        )
        
        dataset2 = GeometricInvarianceDataset(
            records=[sample_record],
            window_size=16,
            window_offset=0,
            fps_scaling=1.0,
            transform=None,
            base_seed=seed,
        )
        
        # Get samples from both datasets
        _, x1 = next(iter(dataset1))
        _, x2 = next(iter(dataset2))
        
        # Check same transformation parameters
        assert x1.attrs["rotation_angle"] == x2.attrs["rotation_angle"]
        assert x1.attrs["flip_horizontal"] == x2.attrs["flip_horizontal"]
        assert x1.attrs["flip_vertical"] == x2.attrs["flip_vertical"]
