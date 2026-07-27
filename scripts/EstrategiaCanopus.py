"""Estrategia Canopus — carrusel de ultimates guiado por el talento de DK.

Al darle START, el bot arranca directamente en la pelea (sin navegación de
menús: la pelea de Demon King con el equipo Canopus se entra a mano) y ejecuta
el ciclo A/B/C del carrusel (ver utilities/canopus_fighting_strategies.py).
Usa la misma infraestructura de farmer que el resto de las estrategias
(login, dailies, botón Update de la GUI, detección de "bot atascado", etc.)
vía ``utilities/canopus_farming_logic.py`` + ``FarmingFactory``.

Modo diagnóstico: ``python EstrategiaCanopus.py --monitor`` no juega nada y
solo reporta orbes + estado del talento (el modo que usamos para calibrar).
"""

import argparse
import time

from utilities.app_config import wait_if_paused
from utilities.canopus_farming_logic import CanopusFarmer, States
from utilities.canopus_orb_reader import CHARACTER_NAMES, read_ally_orbs, read_dk_talent
from utilities.capture_window import capture_window
from utilities.farming_factory import FarmingFactory


def run_monitor():
    print("Estrategia Canopus — monitor de orbes y talento de DK (no juega cartas).")

    def _format_reading(orbs, talent_state, talent_fraction):
        if orbs is None:
            orbs_text = "orbes no legibles"
        else:
            orbs_text = " | ".join(f"{name}: {count}/5" for name, count in zip(CHARACTER_NAMES, orbs))
        if talent_state == "activo":
            talent_text = f"talento DK: ACTIVO (brillo {talent_fraction:.0%})"
        elif talent_state == "desactivado":
            talent_text = f"talento DK: desactivado (brillo {talent_fraction:.0%})"
        else:
            talent_text = "talento DK: no visible"
        return f"{orbs_text}  ||  {talent_text}"

    last_report = None
    while True:
        wait_if_paused()
        try:
            screenshot, _ = capture_window()
        except Exception as e:
            print(f"No pude capturar la ventana del juego: {e}")
            time.sleep(3)
            continue

        orbs = read_ally_orbs(screenshot)
        talent_state, talent_fraction = read_dk_talent(screenshot)
        report_key = (tuple(orbs) if orbs is not None else None, talent_state, round(talent_fraction, 1))
        if report_key != last_report:
            print(_format_reading(orbs, talent_state, talent_fraction))
            last_report = report_key
        time.sleep(1)


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
    parser.add_argument(
        "--monitor",
        action="store_true",
        default=False,
        help="Solo reportar orbes y talento de DK, sin jugar cartas.",
    )
    args = parser.parse_args()

    if args.monitor:
        run_monitor()
        return

    FarmingFactory.main_loop(
        farmer=CanopusFarmer,
        battle_strategy=None,  # Canopus siempre usa CanopusCarouselStrategy internamente
        starting_state=States.FIGHTING,
        max_runs=args.clears,
        password=args.password,
        do_dailies=args.do_dailies,
        do_daily_pvp=args.daily_pvp,
    )


if __name__ == "__main__":
    main()
