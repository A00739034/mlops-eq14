# -*- coding: utf-8 -*-
"""
Módulo de Reproducibilidad: Configuración centralizada de semillas aleatorias.

Este módulo asegura que todas las operaciones aleatorias en el pipeline
utilicen semillas consistentes para garantizar reproducibilidad entre
diferentes entornos y ejecuciones.
"""

import random
import numpy as np
import os
import logging
from typing import Optional

# Semilla global por defecto
DEFAULT_RANDOM_SEED = 42


def set_seed(seed: int = DEFAULT_RANDOM_SEED, verbose: bool = True) -> None:
    """
    Configura todas las semillas aleatorias necesarias para reproducibilidad.
    
    Esta función configura semillas para:
    - Python random
    - NumPy
    - Variables de entorno para TensorFlow/PyTorch (si se usan)
    
    Args:
        seed: Valor de la semilla aleatoria (por defecto: 42)
        verbose: Si es True, imprime información sobre la configuración
    
    Examples:
        >>> set_seed(42)
        Semillas configuradas: Python random, NumPy
        >>> set_seed(42, verbose=False)
    """
    logger = logging.getLogger(__name__)
    
    # Python random
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # Variables de entorno para TensorFlow (si se usa en el futuro)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    
    if verbose:
        logger.info(f"Semillas configuradas a {seed}: Python random, NumPy")
        logger.info(f"Variables de entorno configuradas: PYTHONHASHSEED={seed}")


def set_random_state_for_pandas(pandas_obj, seed: int = DEFAULT_RANDOM_SEED) -> None:
    """
    Configura la semilla aleatoria para operaciones de pandas.
    
    Nota: pandas no tiene una semilla global, pero esta función
    asegura que operaciones aleatorias de pandas usen el estado
    de NumPy que ya fue configurado.
    
    Args:
        pandas_obj: Objeto de pandas (DataFrame, Series, etc.)
        seed: Valor de la semilla (usado para documentación)
    """
    # pandas usa numpy.random internamente, así que si configuramos
    # numpy.random.seed(), pandas ya estará usando esa semilla
    pass


def get_random_state() -> int:
    """
    Obtiene el valor de la semilla aleatoria actual.
    
    Returns:
        Valor de la semilla aleatoria
    """
    return DEFAULT_RANDOM_SEED


class ReproducibilityContext:
    """
    Context manager para garantizar reproducibilidad en bloques de código.
    
    Este context manager puede usarse para asegurar que un bloque específico
    de código use semillas consistentes, incluso si el código externo las cambia.
    
    Examples:
        >>> with ReproducibilityContext(seed=42):
        ...     # Código que debe ser reproducible
        ...     model.fit(X, y)
    """
    
    def __init__(self, seed: int = DEFAULT_RANDOM_SEED):
        """
        Inicializa el context manager.
        
        Args:
            seed: Valor de la semilla aleatoria
        """
        self.seed = seed
        self.prev_random_state = None
        self.prev_numpy_state = None
        self.prev_hashseed = None
    
    def __enter__(self):
        """Entra al contexto y guarda el estado actual."""
        # Guardar estado actual
        self.prev_random_state = random.getstate()
        self.prev_numpy_state = np.random.get_state()
        self.prev_hashseed = os.environ.get('PYTHONHASHSEED')
        
        # Configurar nuevas semillas
        set_seed(self.seed, verbose=False)
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Sale del contexto y restaura el estado anterior."""
        # Restaurar estado anterior
        random.setstate(self.prev_random_state)
        np.random.set_state(self.prev_numpy_state)
        
        if self.prev_hashseed:
            os.environ['PYTHONHASHSEED'] = self.prev_hashseed
        elif 'PYTHONHASHSEED' in os.environ:
            del os.environ['PYTHONHASHSEED']
        
        return False


def configure_reproducibility_for_sklearn(random_state: int = DEFAULT_RANDOM_SEED) -> dict:
    """
    Retorna un diccionario con configuración de random_state para sklearn.
    
    Args:
        random_state: Valor de la semilla aleatoria
        
    Returns:
        Diccionario con configuración para modelos sklearn
    """
    return {
        'random_state': random_state
    }


# Configurar semillas globalmente al importar el módulo
set_seed(DEFAULT_RANDOM_SEED, verbose=False)

