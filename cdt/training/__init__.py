"""
CDT Training Module
"""

from .losses import infonce_loss, ContrastiveLoss
from .trainer import CDTTrainer

__all__ = ['infonce_loss', 'ContrastiveLoss', 'CDTTrainer']
