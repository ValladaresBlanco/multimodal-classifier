"""
Módulo of clasificadores of images
"""

from .resnet_classifier import ResNetClassifier
from .mobilenet_classifier import MobileNetClassifier

__all__ = ['ResNetClassifier', 'MobileNetClassifier']
