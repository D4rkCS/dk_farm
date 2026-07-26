"""Estrategia Canopus — carrusel de ultimates guiado por el talento de DK.

Al darle START, el bot entra directamente a la arquitectura fighter + strategy
(sin capa de farmer): espera a que sea nuestro turno (cartas en mano) y ejecuta
el ciclo A/B/C del carrusel (ver utilities/canopus_fighting_strategies.py).
Al terminar una pelea cierra el resultado, reporta y espera la siguiente.

Modo diagnóstico: ``python EstrategiaCanopus.py --monitor`` no juega nada y
solo reporta orbes + estado del talento (el modo que usamos para calibrar).
"""

import argparse
import time

from utilities.app_config import wait_if_paused
from utilities.canopus_fighter import CanopusFighter
from utilities.canopus_fighting_strategies import CanopusCarouselStrategy
from utilities.canopus_orb_reader import CHARACTER_NAMES, read_ally_orbs, read_dk_talent
from utilities.capture_window import capture_window


class _SessionStats:
    wins = 0
    losses = 0


def _fight_complete(victory=False, **kwargs):
    if victory:
        _SessionStats.wins += 1
        print("¡PELEA GANADA!")
        print("[CLEAR]")
    else:
        _SessionStats.losses += 1
        phase = kwargs.get("phase")
        print(f"Pelea perdida{f' en fase {phase}' if phase is not None else ''}...")
        print("[LOSS]")

    total = _SessionStats.wins + _SessionStats.losses
    print(f"Resultados de la sesión: {_SessionStats.wins}/{total} victorias.")


def run_strategy():
    print("Estrategia Canopus (carrusel) iniciada. Entra a la pelea; actuaré cuando sea tu turno.")

    while True:
        fighter = CanopusFighter(
            battle_strategy=CanopusCarouselStrategy,
            callback=_fight_complete,
        )
        fighter.run()

        # Pequeña pausa antes de buscar la siguiente pelea
        time.sleep(3)


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
    parser.add_argument(
        "--monitor",
        action="store_true",
        default=False,
        help="Solo reportar orbes y talento de DK, sin jugar cartas.",
    )
    args = parser.parse_args()

    if args.monitor:
        run_monitor()
    else:
        run_strategy()


if __name__ == "__main__":
    main()
