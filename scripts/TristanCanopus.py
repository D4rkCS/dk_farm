"""Tristan Canopus — igual que Estrategia Canopus, pero corrige la run sola
cuando el bot comete un error.

Al empezar cada turno revisa si la carta de ulti de Tristan está en la mano:
eso significa que el carrusel se rompió (ver
utilities/tristan_canopus_fighting_strategies.py para el detalle completo de
la secuencia de arreglo). Si no está, juega el turno normal exactamente igual
que Estrategia Canopus -no la reemplaza, la envuelve-.

Script APARTE de EstrategiaCanopus.py a propósito: no modifica el carrusel
principal, así que un error en esta lógica de arreglo no puede corromper la
estrategia principal. Comparte el resto de la infraestructura (login,
dailies, botón Update de la GUI, detección de "bot atascado") vía
utilities/tristan_canopus_farming_logic.py + FarmingFactory.
"""

import argparse

from utilities.canopus_farming_logic import States
from utilities.farming_factory import FarmingFactory
from utilities.tristan_canopus_farming_logic import TristanCanopusFarmer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--password", "-p", type=str, default=None, help="Account password")
    parser.add_argument("--clears", type=str, default="inf", help="How many total clears")
    parser.add_argument("--do-dailies", action="store_true", default=False, help="Do dailies (default: False)")
    parser.add_argument(
        "--daily-pvp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Do daily PVP when dailies run (default: True)",
    )
    args = parser.parse_args()

    FarmingFactory.main_loop(
        farmer=TristanCanopusFarmer,
        battle_strategy=None,  # Tristan Canopus siempre usa TristanCanopusStrategy internamente
        starting_state=States.FIGHTING,
        max_runs=args.clears,
        password=args.password,
        do_dailies=args.do_dailies,
        do_daily_pvp=args.daily_pvp,
    )


if __name__ == "__main__":
    main()
