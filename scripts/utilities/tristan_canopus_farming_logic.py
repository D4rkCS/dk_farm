"""Farmer de Tristan Canopus: igual que ``CanopusFarmer``, pero el fighter
corre ``TristanCanopusStrategy`` (carrusel normal + arreglo automático de
runs rotas) en vez de ``CanopusCarouselStrategy`` directamente.

Script APARTE a propósito: no toca ``canopus_farming_logic.py`` ni
``canopus_fighting_strategies.py``, solo reutiliza toda su infraestructura
por herencia (login, dailies, botón Update, detección de "bot atascado", vía
``CanopusFarmer``/``FarmingFactory``).
"""

from utilities.canopus_farming_logic import CanopusFarmer
from utilities.canopus_fighter import CanopusFighter
from utilities.general_fighter_interface import IFighter
from utilities.tristan_canopus_fighting_strategies import TristanCanopusStrategy


class TristanCanopusFarmer(CanopusFarmer):
    """``CanopusFarmer`` con ``TristanCanopusStrategy`` en el fighter."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.fighter: IFighter = CanopusFighter(
            battle_strategy=TristanCanopusStrategy,
            callback=self.fight_complete_callback,
        )
