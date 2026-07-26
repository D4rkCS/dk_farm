"""Fighter de la Estrategia Canopus (pelea de Demon King con 4 aliados).

Arranca directamente en la arquitectura fighter + strategy, sin capa de
navegación (farmer): al iniciar espera a que sea nuestro turno (slots de carta
vacíos visibles) y juega cartas con ``CanopusBattleStrategy``. Al ver la
pantalla de resultado, la cierra y reporta victoria o derrota por callback.
"""

import time
from typing import Callable

import numpy as np
import utilities.vision_images as vio
from utilities.fighting_strategies import IBattleStrategy
from utilities.general_fighter_interface import FightingStates, IFighter
from utilities.utilities import (
    capture_window,
    find,
    find_and_click,
    get_card_slot_region_image,
)


class CanopusFighter(IFighter):

    def __init__(self, battle_strategy: IBattleStrategy, callback: Callable | None = None):
        super().__init__(battle_strategy=battle_strategy, callback=callback)
        self._victory_seen = False

    def fighting_state(self):
        screenshot, _ = capture_window()

        if find(vio.ok_main_button, screenshot) or find(vio.defeat, screenshot):
            # Terminó la pelea (victoria o derrota); pasar a cerrar el resultado
            self.current_state = FightingStates.EXIT_FIGHT

        elif (available_card_slots := self.count_empty_card_slots(screenshot)) > 0:
            # La mano siempre está visible; la señal de turno es la fila de slots
            # vacíos, que desaparece en el turno enemigo. Confirmar con una segunda
            # lectura para no dispararse en medio de una animación.
            time.sleep(0.5)
            screenshot, _ = capture_window()
            confirmed_slots = self.count_empty_card_slots(screenshot)
            if confirmed_slots <= 0:
                return
            self.available_card_slots = max(available_card_slots, confirmed_slots)
            print(f"MY TURN, selecting {self.available_card_slots} cards...")
            self._apply_detected_phase(self._identify_phase(screenshot))
            self.current_state = FightingStates.MY_TURN

    def _identify_phase(self, screenshot: np.ndarray) -> int | None:
        if find(vio.phase_2, screenshot, threshold=0.8):
            return 2
        if find(vio.phase_3, screenshot, threshold=0.8):
            return 3
        return None

    def my_turn_state(self):
        # El carrusel ejecuta el turno completo de forma imperativa (verificaciones,
        # talento, movimientos, ultimate y reset).
        if hasattr(self.battle_strategy, "execute_turn"):
            self.battle_strategy.execute_turn()
            # Igual que los demás fighters: el turno termina cuando la fila de
            # slots desaparece. Esperar esa señal antes de re-armar la detección,
            # para no volver a ejecutar el guion en el mismo turno.
            self._wait_for_turn_end()
            self.current_state = FightingStates.FIGHTING
            return
        self.play_cards()

    def _wait_for_turn_end(self, timeout: float = 20.0) -> None:
        """Esperar a que los slots desaparezcan (turno ejecutado / turno enemigo).

        Si tras ``timeout`` los slots siguen visibles, volvemos a FIGHTING: la
        detección disparará ``execute_turn`` de nuevo y su lógica de relleno
        completará los movimientos que falten.
        """
        deadline = time.time() + timeout
        while time.time() < deadline:
            screenshot, _ = capture_window()
            if find(vio.ok_main_button, screenshot) or find(vio.defeat, screenshot):
                self.current_state = FightingStates.EXIT_FIGHT
                return
            if self.count_empty_card_slots(screenshot) == 0:
                print("Turno terminado; esperando el siguiente...")
                return
            time.sleep(0.8)
        print("Los slots siguen visibles tras la espera; reintentaré completar el turno.")

    def exit_fight_state(self):
        """Cerrar la pantalla de resultado y reportar el desenlace."""
        screenshot, window_location = capture_window()

        if find(vio.victory, screenshot):
            self._victory_seen = True

        if find_and_click(vio.ok_main_button, screenshot, window_location):
            return

        # Sin botón OK a la vista: el resultado ya se cerró
        victory = self._victory_seen
        with self._lock:
            self.exit_thread = True
        self.complete_callback(victory=victory, phase=IFighter.current_phase)

    @staticmethod
    def count_empty_card_slots(screenshot, threshold=0.6):
        """Contar los slots de carta vacíos (mismos templates que la pelea de DK)."""
        card_slots_image = get_card_slot_region_image(screenshot)

        rectangles, _ = vio.empty_card_slot.find_all_rectangles(card_slots_image, threshold=threshold)
        rectangles_2, _ = vio.empty_card_slot_2.find_all_rectangles(screenshot, threshold=0.7)
        rectangles_3, _ = vio.dk_empty_slot.find_all_rectangles(screenshot, threshold=0.7)

        rectangles = rectangles_3 if rectangles_3.size else rectangles_2 if rectangles_2.size else rectangles

        return 4 if find(vio.skill_locked, screenshot, threshold=0.6) else min(4, len(rectangles))

    @IFighter.run_wrapper
    def run(self):
        print("Estrategia Canopus: esperando nuestro turno (fighter + strategy, sin farmer)...")
        self.prepare_for_new_fight()
        self._victory_seen = False

        while True:

            if self.current_state == FightingStates.FIGHTING:
                self.fighting_state()

            elif self.current_state == FightingStates.MY_TURN:
                self.my_turn_state()

            elif self.current_state == FightingStates.EXIT_FIGHT:
                self.exit_fight_state()

            if self.exit_thread:
                print("Cerrando el fighter de la Estrategia Canopus.")
                return

            time.sleep(0.7)
