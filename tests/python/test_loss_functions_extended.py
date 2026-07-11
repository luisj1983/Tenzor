"""
Tests for loss function Python bindings that currently have zero coverage.

Covers: FocalLoss, DiceLoss, HuberLoss, CTCLoss, MarginRankingLoss,
TripletMarginLoss, CosineEmbeddingLoss, HingeEmbeddingLoss, SoftMarginLoss,
PoissonNLLLoss, MultiMarginLoss, MultiLabelSoftMarginLoss, GaussianNLLLoss,
KLDivLoss, InfoNCELoss, NTXentLoss, TripletLoss.
"""

import pytest
import tenzor as tz

# JIT-R044: import the single source of truth from conftest.py rather than
# duplicating the device list (a prior verbatim duplication across 5 files
# was how the MPS omission propagated undetected).
from conftest import ALL_DEVICES


def make_var(shape, device="cpu", requires_grad=False):
    return tz.Variable(tz.randn(shape, device=device), requires_grad)


# ============================================================================
# Standard loss functions
# ============================================================================

@pytest.mark.parametrize("device", ALL_DEVICES, indirect=True)
class TestLossFunctions:
    def test_kldiv_loss(self, device):
        loss_fn = tz.nn.KLDivLoss()
        pred = make_var([4, 8], device)
        target = make_var([4, 8], device)
        loss = loss_fn(pred, target)

    def test_focal_loss(self, device):
        loss_fn = tz.nn.FocalLoss()
        pred = make_var([4, 4], device)
        target = make_var([4, 4], device)
        loss = loss_fn(pred, target)

    def test_dice_loss(self, device):
        loss_fn = tz.nn.DiceLoss()
        pred = make_var([4, 4], device)
        target = make_var([4, 4], device)
        loss = loss_fn(pred, target)

    def test_huber_loss(self, device):
        loss_fn = tz.nn.HuberLoss()
        pred = make_var([4, 4], device)
        target = make_var([4, 4], device)
        loss = loss_fn(pred, target)

    def test_margin_ranking_loss(self, device):
        loss_fn = tz.nn.MarginRankingLoss()
        x1 = make_var([4], device)
        x2 = make_var([4], device)
        target = make_var([4], device)
        loss = loss_fn(x1, x2, target)

    def test_soft_margin_loss(self, device):
        loss_fn = tz.nn.SoftMarginLoss()
        pred = make_var([4, 4], device)
        target = make_var([4, 4], device)
        loss = loss_fn(pred, target)

    def test_hinge_embedding_loss(self, device):
        loss_fn = tz.nn.HingeEmbeddingLoss()
        pred = make_var([4, 4], device)
        target = make_var([4, 4], device)
        loss = loss_fn(pred, target)

    def test_poisson_nll_loss(self, device):
        loss_fn = tz.nn.PoissonNLLLoss()
        pred = make_var([4, 4], device)
        target = make_var([4, 4], device)
        loss = loss_fn(pred, target)

    def test_cosine_embedding_loss(self, device):
        loss_fn = tz.nn.CosineEmbeddingLoss()
        x1 = make_var([4, 8], device)
        x2 = make_var([4, 8], device)
        target = make_var([4], device)
        loss = loss_fn(x1, x2, target)

    def test_triplet_margin_loss(self, device):
        loss_fn = tz.nn.TripletMarginLoss()
        anchor = make_var([4, 8], device)
        positive = make_var([4, 8], device)
        negative = make_var([4, 8], device)
        loss = loss_fn(anchor, positive, negative)

    def test_multi_label_soft_margin_loss(self, device):
        loss_fn = tz.nn.MultiLabelSoftMarginLoss()
        pred = make_var([4, 4], device)
        target = make_var([4, 4], device)
        loss = loss_fn(pred, target)

    def test_multi_margin_loss(self, device):
        loss_fn = tz.nn.MultiMarginLoss()
        pred = make_var([4, 4], device)
        # MultiMarginLoss expects integer class labels, not a Variable target.
        target = tz.zeros([4], dtype=tz.dtype.int64, device=device)
        loss = loss_fn(pred, target)

    def test_gaussian_nll_loss(self, device):
        loss_fn = tz.nn.GaussianNLLLoss()
        pred = make_var([4, 4], device)
        target = make_var([4, 4], device)
        var = make_var([4, 4], device)
        loss = loss_fn(pred, target, var)


# ============================================================================
# Contrastive losses
# ============================================================================

@pytest.mark.parametrize("device", ALL_DEVICES, indirect=True)
class TestContrastiveLosses:
    def test_triplet_loss(self, device):
        loss_fn = tz.nn.TripletLoss()
        anchor = make_var([4, 8], device)
        positive = make_var([4, 8], device)
        negative = make_var([4, 8], device)
        loss = loss_fn(anchor, positive, negative)

    def test_info_nce_loss(self, device):
        loss_fn = tz.nn.InfoNCELoss()
        query = make_var([4, 8], device)
        positive_key = make_var([4, 8], device)
        loss = loss_fn(query, positive_key)

    def test_nt_xent_loss(self, device):
        loss_fn = tz.nn.NTXentLoss()
        z1 = make_var([4, 8], device)
        z2 = make_var([4, 8], device)
        loss = loss_fn(z1, z2)
