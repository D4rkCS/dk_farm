"""Estrategia "Tristan Canopus": arregla runs rotas de la Estrategia Canopus.

Script APARTE de ``canopus_fighting_strategies.py`` a propósito (no lo toca
ni lo modifica): ``TristanCanopusStrategy`` hereda de ``CanopusCarouselStrategy``
y juega el carrusel normal sin ningún cambio, pero antes de cada turno revisa
si la ulti de Tristan está en la mano. Si no está, delega el turno completo a
la estrategia base (``super().execute_turn()``). Si está, es la señal de que
el carrusel se rompió -alguna ultimate de dk/cusack/galand se quedó sin poder
confirmarse y el turno se consumió sin liberar el lugar en la mano, así que
Tristan (que no participa del carrusel normal) terminó con la suya lista- y
en vez de jugar el turno normal corre la secuencia de arreglo:

  1. Cargar (VERIFICADO, con el mismo patrón de mover+verificar+resetear que
     el resto del bot) a los 2 personajes del carrusel más cerca de sus 5
     orbes, hasta el tope. Como un turno normal solo tiene 4 slots, esto
     puede abarcar más de un turno: cada vez que ``execute_turn`` se llama de
     nuevo con la ulti de Tristan todavía en mano, retoma desde los orbes
     reales (ya cargados) en vez de repetir movimientos de más.
  2. Con los 2 personajes verificados a 5, jugar la carta de ulti de Tristan.
  3. Saltar (clickear) los slots de carta que hayan quedado vacíos este
     turno para pasarlo sin arriesgar más jugadas: clickear un slot todavía
     vacío (el mismo ícono, ``empty_card_slot.png`` y compañía, que usa todo
     el bot para saber si es nuestro turno) lo salta directamente, sin
     necesidad de jugar nada ahí.
  4. Turno(s) siguientes: mover cartas de Tristan 6 veces en total (puede
     abarcar más de un turno, igual que el paso 1). Cada movimiento se
     confirma contra los slots del turno, no contra la barra de orbes de
     Tristan (que en esta fase a veces queda tapada por animaciones y daba
     falsos negativos que reseteaban movimientos válidos). Una vez
     completados los 6, saltar los slots que queden vacíos para pasar el
     turno.
  5. Run arreglada: se apaga la fase de arreglo y el próximo turno vuelve al
     carrusel normal. Si en el futuro el carrusel se rompe de nuevo, la ulti
     de Tristan va a volver a aparecer en la mano y el ciclo se repite solo.
"""

import time

import utilities.vision_images as vio
from utilities.app_config import wait_if_paused
from utilities.canopus_fighting_strategies import (
    _MAX_ATTEMPTS,
    CHARACTER_NAMES,
    FOCUS_ORDER,
    ULT_TEMPLATES,
    CanopusCarouselStrategy,
)
from utilities.capture_window import capture_window
from utilities.coordinates import Coordinates
from utilities.utilities import click_im, drag_im, find, get_card_slot_region_image, get_click_point_from_rectangle

# Cuántas cartas de Tristan mover en la fase de recarga, tras jugar su ulti.
_TRISTAN_RECHARGE_MOVES = 6

# Cuánto esperar (reintentando) a que aparezca el ícono de "slot movido con
# éxito" tras un movimiento de carta de Tristan: es transitorio, así que hay
# que sondear en vez de mirar una sola vez.
_MOVE_SUCCESS_ICON_TIMEOUT = 2.0
_MOVE_SUCCESS_ICON_POLL = 0.15


def _empty_slot_rectangles(screenshot):
    """Rectángulos ``[x, y, w, h]`` de los slots VACÍOS, en coordenadas del
    screenshot completo. Misma prioridad de plantillas que
    ``CanopusFighter.count_empty_card_slots`` (dk_empty_slot >
    empty_card_slot_2 > empty_card_slot), pero devolviendo los rectángulos ya
    trasladados a coordenadas del screenshot completo: los dos primeros ya
    vienen así (matchean contra el screenshot entero); el tercero matchea
    contra el recorte de ``card_slots_region``, así que hay que sumarle su
    offset.
    """
    rectangles_3, _ = vio.dk_empty_slot.find_all_rectangles(screenshot, threshold=0.7)
    if rectangles_3.size:
        return rectangles_3

    rectangles_2, _ = vio.empty_card_slot_2.find_all_rectangles(screenshot, threshold=0.7)
    if rectangles_2.size:
        return rectangles_2

    region = Coordinates.get_coordinates("card_slots_region")
    card_slots_image = get_card_slot_region_image(screenshot)
    rectangles, _ = vio.empty_card_slot.find_all_rectangles(card_slots_image, threshold=0.6)
    if rectangles.size:
        rectangles = rectangles.copy()
        rectangles[:, 0] += region[0]
        rectangles[:, 1] += region[1]
    return rectangles


class TristanCanopusStrategy(CanopusCarouselStrategy):
    """Carrusel normal (heredado sin cambios) + arreglo automático de runs rotas."""

    def __init__(self):
        super().__init__()
        # True mientras estemos en la fase de recarga de Tristan tras jugar
        # su ulti (ver _continue_recharge). Persiste turno a turno.
        self._fixing_recharge: bool = False
        # Cuántos de los 6 movimientos de recarga de Tristan ya se verificaron.
        self._recharge_moves_done: int = 0

    # ── Punto de entrada del turno ──

    def execute_turn(self) -> None:
        wait_if_paused()

        if self._fixing_recharge:
            self._continue_recharge()
            return

        tristan_card, tristan_score = self._char_ult_in_hand("tristan")
        if tristan_card is None:
            # Sin señal de error: turno normal del carrusel, sin ningún cambio.
            super().execute_turn()
            return

        print(
            f"[Tristan] Ulti de Tristan en mano (score {tristan_score:.2f}): "
            "el carrusel se rompió. Arreglando la run..."
        )
        self._fix_run(tristan_score)

    # ── Paso 1+2+3: cargar 2 personajes a 5, jugar la ulti de Tristan, pasar turno ──

    def _fix_run(self, tristan_score: float) -> None:
        orbs = self._read_orbs_reliable()
        if orbs is None:
            print("[Tristan] No puedo leer los orbes para el arreglo; reintento el próximo turno.")
            return

        if not self._charge_two_to_full(orbs):
            print("[Tristan] Carga de los 2 personajes sin completar; reintento el próximo turno.")
            return

        if not self._play_specific_ult("tristan", tristan_score):
            print("[Tristan] No pude confirmar que se jugó la ulti de Tristan; reintento el próximo turno.")
            return

        print("[Tristan] Ulti de Tristan jugada. Salto lo que quede vacío para pasar el turno.")
        self._skip_remaining_slots()
        self._fixing_recharge = True
        self._recharge_moves_done = 0

    def _charge_two_to_full(self, orbs: list[int]) -> bool:
        """Carga (VERIFICADO) a los 2 personajes del carrusel más cerca de
        sus 5 orbes, hasta el tope. Igual que el resto del carrusel, el plan
        (a quién cargar y cuánto) se fija con la lectura de orbes de entrada:
        los reintentos tras un reset repiten el mismo plan, no vuelven a
        elegir. Si se acaban los slots del turno a mitad de camino (no es un
        error, un turno normal solo tiene 4), corta sin resetear -el progreso
        ya está verificado y se retoma solo el próximo turno-.
        """
        order = sorted(
            range(len(CHARACTER_NAMES)),
            key=lambda i: (-orbs[i], FOCUS_ORDER.index(CHARACTER_NAMES[i])),
        )
        for idx in order[:2]:
            char = CHARACTER_NAMES[idx]
            needed = 5 - orbs[idx]
            if needed <= 0:
                print(f"[Tristan] {char} ya tiene 5 orbes.")
                continue
            print(f"[Tristan] Plan: cargar {needed} carta(s) de {char} ({orbs[idx]} orbes) hasta 5.")

            verified = False
            for attempt in range(1, _MAX_ATTEMPTS + 1):
                if self._empty_slots() <= 0:
                    print("[Tristan] Sin slots vacíos a mitad de la carga; continúo el próximo turno.")
                    return False

                moved = self._move_char_cards(char, needed)
                after = self._read_orbs_reliable()
                if moved == needed and after is not None and after[idx] >= 5:
                    print(f"[Tristan] Verificado: {char} quedó con {after[idx]} orbes.")
                    verified = True
                    break

                if self._empty_slots() <= 0:
                    print(f"[Tristan] Slots agotados a mitad de la carga de {char}; continúo el próximo turno.")
                    return False

                got = "?" if after is None else after[idx]
                print(
                    f"[Tristan] Carga de {char} sin verificar (movidas {moved}/{needed}, quedó con {got}); "
                    "reseteo y reintento el mismo plan."
                )
                if attempt < _MAX_ATTEMPTS:
                    self._press_reset()

            if not verified:
                print(f"[Tristan] No pude verificar a {char} con 5 orbes tras {_MAX_ATTEMPTS} intentos.")
                return False
        return True

    def _skip_remaining_slots(self, max_rounds: int = 6) -> None:
        """Saltea los slots de carta que hayan quedado vacíos este turno,
        para pasarlo sin jugar el resto de las cartas.

        Clickear un slot todavía vacío (el mismo ícono, ``empty_card_slot.png``
        y compañía, que usa todo el bot para saber si es nuestro turno) lo
        salta directamente -no hace falta "cancelar" nada ni tocar los que ya
        tienen una jugada asignada-. Se repite (redetectando en cada vuelta,
        porque clickear uno puede correr la posición de los que quedan) hasta
        que no queden vacíos -turno pasado- o se agoten los intentos.
        """
        for _ in range(max_rounds):
            if self._empty_slots() <= 0:
                print("[Tristan] Turno pasado (sin slots vacíos).")
                return

            screenshot, window_location = capture_window()
            empty_rects = _empty_slot_rectangles(screenshot)
            if not len(empty_rects):
                print("[Tristan] No encuentro slots vacíos para saltar; sigo de todas formas.")
                return

            for x, y, w, h in empty_rects:
                wait_if_paused()
                point = (int(x + w / 2), int(y + h / 2))
                click_im(point, window_location, sleep_after_click=0.2)
            time.sleep(0.6)

        print("[Tristan] Seguimos viendo slots vacíos tras varios intentos; sigo de todas formas.")

    # ── Paso 4: recarga de Tristan tras jugar su ulti ──

    def _continue_recharge(self) -> None:
        remaining = _TRISTAN_RECHARGE_MOVES - self._recharge_moves_done
        print(
            f"[Tristan] Fase de recarga: {self._recharge_moves_done}/{_TRISTAN_RECHARGE_MOVES} "
            f"movimiento(s) de Tristan hechos, faltan {remaining}."
        )

        for attempt in range(1, _MAX_ATTEMPTS + 1):
            if self._empty_slots() <= 0:
                print("[Tristan] Sin slots vacíos; sigo la recarga el próximo turno.")
                return

            moved, ran_out_of_slots = self._move_tristan_cards(remaining)

            if ran_out_of_slots:
                self._recharge_moves_done += moved
                print(
                    f"[Tristan] Slots agotados: {moved} movimiento(s) verificados este turno "
                    f"({self._recharge_moves_done}/{_TRISTAN_RECHARGE_MOVES} en total). Sigo el próximo turno."
                )
                return

            if moved >= remaining:
                self._recharge_moves_done += moved
                print(f"[Tristan] {_TRISTAN_RECHARGE_MOVES} movimientos completados. Salto lo que quede del turno.")
                self._skip_remaining_slots()
                self._fixing_recharge = False
                self._recharge_moves_done = 0
                print("[Tristan] Run arreglada; el próximo turno retoma el carrusel normal.")
                return

            print(
                f"[Tristan] Movimiento de Tristan sin verificar (solo {moved}/{remaining} este intento); "
                "reseteo y reintento el mismo plan."
            )
            if attempt < _MAX_ATTEMPTS:
                self._press_reset()

        print("[Tristan] Agotados los intentos de recarga este turno; sigo el próximo.")

    def _wait_for_move_success_icon(self) -> bool:
        """Sondea (hasta ``_MOVE_SUCCESS_ICON_TIMEOUT``) el ícono de "slot
        movido con éxito" sobre la fila de slots. Es transitorio -aparece y
        se apaga solo-, así que hay que reintentar en vez de mirar una sola
        vez con un único screenshot.
        """
        deadline = time.time() + _MOVE_SUCCESS_ICON_TIMEOUT
        while time.time() <= deadline:
            screenshot, _ = capture_window()
            card_slots_image = get_card_slot_region_image(screenshot)
            if find(vio.canopus_slot_movido_exito, card_slots_image, threshold=0.7):
                return True
            time.sleep(_MOVE_SUCCESS_ICON_POLL)
        return False

    def _move_tristan_cards(self, times: int) -> tuple[int, bool]:
        """Mueve hasta ``times`` cartas de Tristan, confirmando cada una con
        el ícono de "slot movido con éxito" (``canopus_slot_movido_exito``)
        que aparece brevemente tras un movimiento válido, o -si no se llega a
        ver a tiempo- con los slots del turno (la misma señal de "se jugó una
        carta" que usa el resto del bot). Ninguna de las dos depende de la
        barra de orbes de Tristan: a veces queda tapada por animaciones
        durante esta fase, y una lectura fallida ahí no debería tirar abajo
        movimientos que sí se hicieron. Nunca elige su propia carta de ulti
        como origen (arrastrarla no la mueve).

        Devuelve ``(movimientos confirmados, si se cortó por falta de
        slots)``: esa segunda señal es la que usa ``_continue_recharge`` para
        distinguir "se acabó el turno" (progreso real, seguir el próximo)
        de "un movimiento no se pudo confirmar" (hay que resetear y
        reintentar el mismo plan).
        """
        moved = 0
        for _ in range(times):
            wait_if_paused()
            slots_before = self._empty_slots()
            if slots_before <= 0:
                print("[Tristan] Sin slots vacíos; corto el movimiento de Tristan.")
                return moved, True

            snapshot = self._hand_snapshot()
            own = [
                (pos, item)
                for pos, item in enumerate(snapshot)
                if item[2] == "tristan" and item[3] != ULT_TEMPLATES["tristan"]
            ]
            if not own:
                print("[Tristan] No encuentro cartas de Tristan en la mano para mover.")
                return moved, False

            own.sort(key=lambda pair: -pair[1][4])
            _origin_idx, origin_card, _c, _t, _score = own[0][1]
            target_card = self._pick_move_target(snapshot, own[0][0])
            if target_card is None:
                print("[Tristan] No hay carta objetivo para el movimiento.")
                return moved, False

            _, window_location = capture_window()
            origin_point = get_click_point_from_rectangle(origin_card.rectangle)
            target_point = get_click_point_from_rectangle(target_card.rectangle)
            drag_im(origin_point, target_point, window_location, sleep_after_click=0.1, drag_duration=0.25)

            icon_seen = self._wait_for_move_success_icon()
            slots_after = self._empty_slots()
            if not icon_seen and slots_after >= slots_before:
                print(
                    f"[Tristan] Movimiento no confirmado (sin ícono de éxito, slots antes={slots_before}, "
                    f"después={slots_after}); corto el bloque."
                )
                return moved, False

            moved += 1

        return moved, False
