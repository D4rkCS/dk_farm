"""Estrategia "carrusel" de la Estrategia Canopus (Demon King, 4 aliados).

Ciclo infinito de 3 turnos guiado por el estado del talento de DK. Solo se
mueven cartas para cargar orbes y se juegan las ultimates que van apareciendo.
Tristan (4to aliado, última posición a la derecha) queda fuera del carrusel:
el bot no detecta sus orbes ni mueve sus cartas, solo actúa sobre DK, Cusack
y Galand.

  Turno A (dk_talento ACTIVO):
    1. Click al talento de DK.
    2. Mover 3 cartas del personaje con MENOS orbes.
    3. Verificar que sus orbes subieron +3; si no, btn_reset y rehacer.
    4. Jugar la carta de ultimate de la mano.
    -> el talento pasa a la imagen dk_talento_desactivado_1.

  Turno B (dk_talento_desactivado_1):
    1. Mover 2 cartas del personaje con 3 orbes y 1 del personaje con 0 orbes.
    2. Verificar; si no, btn_reset y rehacer.
    3. Jugar la ultimate de la mano.
    -> el talento pasa a la imagen dk_talento_desactivado_2.

  Turno C (dk_talento_desactivado_2):
    1. Mover 4 veces cartas del personaje con 1 orbe. Sin ultimate.
    -> fin del ciclo; el siguiente turno el talento vuelve a estar ACTIVO.

Al inicio de cada turno (solo cuando hay cartas en mano) se hacen las 3
verificaciones: estado del talento, orbes de cada personaje y ultimate en mano.

Identificación de estados del talento: "activo" por brillo (calibrado y
verificado en vivo). Las etapas _1/_2 no se distinguen bien por template
matching (misma calavera), así que se usa la secuencia del ciclo (tras A viene
B, tras B viene C) y, en arranque en frío, la fracción de púrpura vivo
calibrada con las capturas del usuario (_1 ~4%, _2 ~17%, base ~31%).
"""

import os
import time
from functools import lru_cache

import cv2
import numpy as np
from utilities.app_config import wait_if_paused
from utilities.canopus_orb_reader import (
    CHARACTER_NAMES,
    read_ally_orbs,
    read_dk_talent_metrics,
)
from utilities.capture_window import capture_window
from utilities.card_data import Card, CardTypes
from utilities.fighting_strategies import IBattleStrategy, SmarterBattleStrategy
from utilities.utilities import (
    click_im,
    drag_im,
    get_click_point_from_rectangle,
    get_hand_cards,
)

_IMG_DIR = os.path.join("images", "canopus_dk")

# Templates de cartas por personaje (la ulti es siempre el último nombre).
# Tristan queda fuera a propósito: el bot no debe mover sus cartas ni leer
# sus orbes, así que no se le asigna template ni entra en FOCUS_ORDER.
CHAR_TEMPLATES = {
    "dk": ("dk_single", "dk_area", "dk_ulti"),
    "cusack": ("cusack_single", "cusack_orbe", "cusack_ulti"),
    "galand": ("galand_single", "galand_desventaja", "galand_ulti"),
}
ULT_TEMPLATES = {
    "dk": "dk_ulti",
    "cusack": "cusack_ulti",
    "galand": "galand_ulti",
}
# Prioridad en empates: el foco de la estrategia es DK, Cusack y Galand.
FOCUS_ORDER = ("dk", "cusack", "galand")

_MIN_CARD_SCORE = 0.35  # score mínimo para asignar una carta a un personaje
_MIN_ULT_SCORE = 0.40  # score mínimo para dar por encontrada una ulti en mano
_ULT_VERIFY_SCORE = 0.32  # umbral (más bajo) para "la ulti sigue en mano" tras el click
_MAX_ATTEMPTS = 5  # reintentos de un guion (con btn_reset entre intentos)
_MOVE_SLEEP = 1.2  # cooldown tras cada movimiento de carta (el juego necesita registrarlo)
_PRE_CLICK_SLEEP = 1.0  # cooldown antes de clickear una carta (p. ej. la ultimate)
_ULT_SLEEP = 1.5  # espera tras jugar una ultimate
_TALENT_SLEEP = 2.5  # espera tras click al talento
_POST_TALENT_SLEEP = 1.0  # cooldown extra tras confirmar el talento, antes de empezar a mover cartas

# Punto del botón de talento (calibrado; escala estándar 550x940)
_TALENT_POINT = (275, 657)

# Frontera de vivid_fraction para arranque en frío: _1 (~4%) vs _2 (~17%)
_STAGE_1_VIVID_MAX = 0.10
_STAGE_2_VIVID_MAX = 0.24


@lru_cache(maxsize=None)
def _load_template(name: str) -> np.ndarray | None:
    return cv2.imread(os.path.join(_IMG_DIR, f"{name}.png"))


def _match_scaled(image: np.ndarray, template: np.ndarray) -> float:
    """Mejor score de template matching reescalando el template al ancho de la imagen.

    Los templates del usuario vienen a dos escalas (la primera carta de la mano
    se muestra más grande), así que se prueba el ajuste al ancho y variantes.
    """
    ih, iw = image.shape[:2]
    th, tw = template.shape[:2]
    best = 0.0
    for factor in (0.95, 0.85, 0.75, 1.0):
        scale = iw * factor / tw
        resized = cv2.resize(template, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        resized = resized[: min(resized.shape[0], ih), : min(resized.shape[1], iw)]
        if resized.shape[0] < 10 or resized.shape[1] < 10:
            continue
        result = cv2.matchTemplate(image, resized, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)
        best = max(best, float(max_val))
    return best


class CanopusCarouselStrategy(IBattleStrategy):
    """Ejecuta el carrusel completo de un turno de forma imperativa."""

    def __init__(self):
        # Etapa esperada para el próximo turno desactivado: "B", "C" o None (esperar A/inferir).
        self._next_stage: str | None = None

    # ── Verificaciones ──

    def _read_orbs_reliable(self) -> list[int] | None:
        """Leer orbes con reintentos (animaciones pueden tapar barras)."""
        for _ in range(5):
            screenshot, _ = capture_window()
            orbs = read_ally_orbs(screenshot)
            if orbs is not None:
                return orbs
            time.sleep(0.5)
        return None

    def _identify_card(self, card: Card) -> tuple[str | None, str | None, float]:
        """(personaje, template, score) de la carta, o (None, None, 0) si no matchea."""
        if card.card_image is None:
            return None, None, 0.0
        best_char, best_tpl, best_score = None, None, 0.0
        for char, names in CHAR_TEMPLATES.items():
            for name in names:
                tpl = _load_template(name)
                if tpl is None:
                    continue
                score = _match_scaled(card.card_image, tpl)
                if score > best_score:
                    best_char, best_tpl, best_score = char, name, score
        if best_score < _MIN_CARD_SCORE:
            return None, None, best_score
        return best_char, best_tpl, best_score

    def _hand_snapshot(self) -> list[tuple[int, Card, str | None, str | None, float]]:
        """[(idx, carta, personaje, template, score)] de las cartas no-suelo de la mano."""
        hand = get_hand_cards()
        snapshot = []
        for idx, card in enumerate(hand):
            if card.card_type in (CardTypes.GROUND, CardTypes.NONE):
                continue
            char, tpl, score = self._identify_card(card)
            snapshot.append((idx, card, char, tpl, score))
        return snapshot

    def _find_ult_in_hand(self) -> tuple[Card | None, str | None, float]:
        """La mejor carta de ultimate presente en la mano: (carta, personaje, score).

        Una carta solo cuenta como ultimate si el template de ulti es su MEJOR
        match: las cartas normales de un personaje pueden matchear su template
        de ulti como match secundario (p. ej. cusack_single da ~0.43 contra
        cusack_ulti) y eso produciría falsos positivos.
        """
        best_card, best_char, best_score = None, None, 0.0
        for _idx, card, char, tpl, score in self._hand_snapshot():
            if char is None or tpl != ULT_TEMPLATES[char]:
                continue
            if score > best_score:
                best_card, best_char, best_score = card, char, score
        if best_score < _MIN_ULT_SCORE:
            return None, None, best_score
        return best_card, best_char, best_score

    def _classify_stage(self, metrics: dict, orbs: list[int] | None, has_ult: bool) -> str | None:
        """Determina el turno del ciclo: "A", "B", "C" o None si no hay señal."""
        if metrics["estado"] == "activo":
            return "A"
        if metrics["estado"] == "no_visible":
            return None

        # Desactivado: primero la secuencia del ciclo, luego inferencia en frío.
        if self._next_stage in ("B", "C"):
            return self._next_stage

        vivid = metrics["vivid_fraction"]
        if vivid < _STAGE_1_VIVID_MAX:
            return "B"  # imagen dk_talento_desactivado_1
        if vivid < _STAGE_2_VIVID_MAX:
            return "C"  # imagen dk_talento_desactivado_2

        # Desactivado "base": inferir por la firma de orbes
        if orbs is not None:
            if 3 in orbs and has_ult:
                return "B"
            if 1 in orbs:
                return "C"
        return None

    # ── Acciones básicas ──

    def _empty_slots(self) -> int:
        from utilities.canopus_fighter import CanopusFighter

        screenshot, _ = capture_window()
        return CanopusFighter.count_empty_card_slots(screenshot)

    def _click_talent(self) -> None:
        """Click al talento de DK, verificando que el botón deja de brillar.

        Tras confirmarlo espera un segundo extra antes de devolver el control,
        para que el juego asiente antes de empezar a mover cartas.
        """
        for attempt in range(1, _MAX_ATTEMPTS + 1):
            _, window_location = capture_window()
            print(f"[Turno A] Click al talento de DK (intento {attempt}/{_MAX_ATTEMPTS}).")
            click_im(_TALENT_POINT, window_location)
            time.sleep(_TALENT_SLEEP)
            screenshot, _ = capture_window()
            if read_dk_talent_metrics(screenshot)["estado"] != "activo":
                time.sleep(_POST_TALENT_SLEEP)
                return
        print(f"[Turno A] El talento sigue brillando tras {_MAX_ATTEMPTS} clicks; continúo de todas formas.")
        time.sleep(_POST_TALENT_SLEEP)

    def _press_reset(self) -> bool:
        """Click al botón RESET de la pelea (deshace los movimientos del turno)."""
        template = _load_template("btn_reset")
        if template is None:
            print("No encuentro images/canopus_dk/btn_reset.png; no puedo resetear.")
            return False
        screenshot, window_location = capture_window()
        best = None
        for scale in (1.0, 0.86, 0.75, 1.15):
            resized = cv2.resize(template, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            result = cv2.matchTemplate(screenshot, resized, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)
            if best is None or max_val > best[0]:
                best = (max_val, max_loc, resized.shape)
        score, loc, shape = best
        if score < 0.6:
            print(f"Botón RESET no visible (score {score:.2f}); nada que deshacer.")
            return False
        center = (loc[0] + shape[1] // 2, loc[1] + shape[0] // 2)
        print(f"Pulsando RESET para deshacer los movimientos (score {score:.2f}).")
        click_im(center, window_location)
        time.sleep(1.2)
        return True

    def _pick_move_target(self, snapshot, origin_pos: int) -> Card | None:
        """Carta ocupada más cercana al origen; evita merges (mismo personaje y rango) si puede."""
        origin_idx, origin_card, origin_char, _tpl, _s = snapshot[origin_pos]
        candidates = sorted(
            (item for pos, item in enumerate(snapshot) if pos != origin_pos),
            key=lambda item: abs(item[0] - origin_idx),
        )
        for _idx, card, char, _tpl2, _s2 in candidates:
            if char == origin_char and card.card_rank == origin_card.card_rank:
                # Posible merge (mismo personaje y rango): probar otro objetivo
                continue
            return card
        # Sin objetivo "seguro": usar el más cercano de todas formas
        return candidates[0][1] if candidates else None

    def _move_char_cards(self, char: str, times: int) -> int:
        """Mueve ``times`` cartas del personaje dado. Devuelve cuántos movimientos hizo."""
        moved = 0
        for _ in range(times):
            wait_if_paused()
            snapshot = self._hand_snapshot()
            # Candidatas del personaje, prefiriendo cartas que no sean su ulti
            own = [
                (pos, item)
                for pos, item in enumerate(snapshot)
                if item[2] == char
            ]
            if not own:
                print(f"No encuentro cartas de {char} en la mano para mover.")
                break
            own.sort(key=lambda pair: (pair[1][3] == ULT_TEMPLATES[char], -pair[1][4]))
            origin_pos, (origin_idx, origin_card, _c, _t, score) = own[0]
            target_card = self._pick_move_target(snapshot, origin_pos)
            if target_card is None:
                print("No hay carta objetivo para el movimiento.")
                break
            _, window_location = capture_window()
            origin_point = get_click_point_from_rectangle(origin_card.rectangle)
            target_point = get_click_point_from_rectangle(target_card.rectangle)
            print(f"Moviendo carta de {char} (idx {origin_idx}, score {score:.2f}).")
            drag_im(origin_point, target_point, window_location, sleep_after_click=0.1, drag_duration=0.25)
            time.sleep(_MOVE_SLEEP)
            moved += 1
        return moved

    def _char_ult_in_hand(self, char: str) -> tuple[Card | None, float]:
        """La carta de ultimate de ``char`` si sigue en la mano: (carta, score).

        Igual que en ``_find_ult_in_hand``, solo cuenta si el template de ulti
        es el MEJOR match de la carta. Para la verificación post-click se usa
        un umbral algo más bajo: ante la duda, mejor asumir que sigue en mano
        y reintentar el click (siempre sobre la posición fresca de la carta).
        """
        for _idx, card, c, tpl, score in self._hand_snapshot():
            if c == char and tpl == ULT_TEMPLATES[char] and score >= _ULT_VERIFY_SCORE:
                return card, score
        return None, 0.0

    def _play_ult(self) -> bool:
        """Juega la mejor ultimate de la mano, verificando que desaparece tras el click."""
        card, char, score = self._find_ult_in_hand()
        if card is None:
            print(f"No hay ultimate reconocible en la mano (mejor score {score:.2f}).")
            return False

        for attempt in range(1, _MAX_ATTEMPTS + 1):
            _, window_location = capture_window()
            time.sleep(_PRE_CLICK_SLEEP)
            print(f"Jugando la ultimate de {char} (score {score:.2f}, intento {attempt}/{_MAX_ATTEMPTS}).")
            click_im(card.rectangle, window_location)
            time.sleep(_ULT_SLEEP)

            # Verificar que la ultimate ya no está en la mano (el click pudo perderse)
            card, score = self._char_ult_in_hand(char)
            if card is None:
                print(f"Verificado: la ultimate de {char} desapareció de la mano.")
                return True
            print(f"La ultimate de {char} sigue en la mano (score {score:.2f}); reintento el click.")

        print(f"No pude confirmar que la ultimate de {char} se jugara tras {_MAX_ATTEMPTS} clicks.")
        return False

    def _char_by_orbs(self, orbs: list[int], predicate, label: str) -> int | None:
        """Índice del personaje cuyo conteo cumple ``predicate``; empates por FOCUS_ORDER."""
        matching = [i for i, count in enumerate(orbs) if predicate(count)]
        if not matching:
            print(f"Ningún personaje cumple: {label}. Orbes: {orbs}")
            return None
        matching.sort(key=lambda i: FOCUS_ORDER.index(CHARACTER_NAMES[i]))
        return matching[0]

    # ── Guiones de turno ──

    def _turn_a(self, orbs: list[int]) -> None:
        print(f"[Turno A] Talento ACTIVO. Orbes: {self._orbs_text(orbs)}")
        self._click_talent()

        current = orbs
        for attempt in range(1, _MAX_ATTEMPTS + 1):
            target = min(range(len(current)), key=lambda i: (current[i], FOCUS_ORDER.index(CHARACTER_NAMES[i])))
            char = CHARACTER_NAMES[target]
            before = current[target]
            expected = min(5, before + 3)
            print(f"[Turno A] Moviendo 3 cartas de {char} ({before} orbes; espero {expected}).")
            moved = self._move_char_cards(char, 3)

            after = self._read_orbs_reliable()
            if moved == 3 and after is not None and after[target] >= expected:
                print(f"[Turno A] Verificado: {char} quedó con {after[target]} orbes.")
                if self._play_ult():
                    self._next_stage = "B"
                    return
                print("[Turno A] No pude jugar la ultimate.")
            else:
                got = "?" if after is None else after[target]
                print(f"[Turno A] Verificación fallida (movidas {moved}, {char} con {got} orbes).")

            if attempt < _MAX_ATTEMPTS:
                self._press_reset()
                current = self._read_orbs_reliable() or current

        print("[Turno A] Agotados los intentos; termino el turno como pueda.")
        self._next_stage = "B"

    def _turn_b(self, orbs: list[int]) -> None:
        print(f"[Turno B] Talento en cooldown (etapa 1). Orbes: {self._orbs_text(orbs)}")

        current = orbs
        for attempt in range(1, _MAX_ATTEMPTS + 1):
            idx3 = self._char_by_orbs(current, lambda n: n == 3, "tener 3 orbes")
            idx0 = self._char_by_orbs(current, lambda n: n == 0, "tener 0 orbes")
            if idx3 is None:
                idx3 = self._char_by_orbs(current, lambda n: n < 5, "tener menos de 5 orbes (fallback)")
            if idx0 is None or idx0 == idx3:
                idx0 = self._char_by_orbs(
                    current, lambda n: n == min(current), "tener los menos orbes (fallback)"
                )
            if idx3 is None or idx0 is None:
                break

            char3, char0 = CHARACTER_NAMES[idx3], CHARACTER_NAMES[idx0]
            before3, before0 = current[idx3], current[idx0]
            print(f"[Turno B] 2 movimientos de {char3} ({before3} orbes) y 1 de {char0} ({before0} orbes).")
            moved3 = self._move_char_cards(char3, 2)
            moved0 = self._move_char_cards(char0, 1)

            after = self._read_orbs_reliable()
            ok = (
                moved3 == 2
                and moved0 == 1
                and after is not None
                and after[idx3] >= min(5, before3 + 2)
                and after[idx0] >= min(5, before0 + 1)
            )
            if ok:
                print(f"[Turno B] Verificado: {char3}={after[idx3]}, {char0}={after[idx0]} orbes.")
                if self._play_ult():
                    self._next_stage = "C"
                    return
                print("[Turno B] No pude jugar la ultimate.")
            else:
                print(f"[Turno B] Verificación fallida (movidas {moved3}+{moved0}, orbes {after}).")

            if attempt < _MAX_ATTEMPTS:
                self._press_reset()
                current = self._read_orbs_reliable() or current

        print("[Turno B] Agotados los intentos; termino el turno como pueda.")
        self._next_stage = "C"

    def _turn_c(self, orbs: list[int]) -> None:
        print(f"[Turno C] Talento en cooldown (etapa 2). Orbes: {self._orbs_text(orbs)} — sin ultimate.")

        idx1 = self._char_by_orbs(orbs, lambda n: n == 1, "tener 1 orbe")
        if idx1 is None:
            idx1 = self._char_by_orbs(orbs, lambda n: n == min(orbs), "tener los menos orbes (fallback)")
        if idx1 is not None:
            char = CHARACTER_NAMES[idx1]
            print(f"[Turno C] Moviendo 4 cartas de {char} ({orbs[idx1]} orbes).")
            self._move_char_cards(char, 4)

        # Cierra el ciclo: el siguiente turno debería volver a tener el talento ACTIVO.
        self._next_stage = None

    def _filler_turn(self, orbs: list[int] | None) -> None:
        """Sin señal clara del ciclo: mover cartas del personaje con menos orbes para no estancarse."""
        print("[Turno ?] Sin señal clara del ciclo; muevo cartas del personaje con menos orbes.")
        if orbs is None:
            orbs = [0] * len(CHARACTER_NAMES)
        target = min(range(len(orbs)), key=lambda i: (orbs[i], FOCUS_ORDER.index(CHARACTER_NAMES[i])))
        self._move_char_cards(CHARACTER_NAMES[target], 2)

    def _ensure_turn_finished(self) -> None:
        """Rellenar los slots que falten con movimientos del personaje con menos orbes.

        Si no encuentra cartas de ese personaje para mover, no hace ningún
        movimiento a ciegas: prefiere dejar el turno como esté a arriesgar una
        jugada al azar.
        """
        for _ in range(6):
            wait_if_paused()
            if self._empty_slots() == 0:
                return
            orbs = self._read_orbs_reliable() or [0] * len(CHARACTER_NAMES)
            target = min(range(len(orbs)), key=lambda i: (orbs[i], FOCUS_ORDER.index(CHARACTER_NAMES[i])))
            if self._move_char_cards(CHARACTER_NAMES[target], 1) == 0:
                print("No encuentro cartas para completar el turno; no arriesgo un movimiento al azar.")
                return
            time.sleep(0.4)

    @staticmethod
    def _orbs_text(orbs: list[int] | None) -> str:
        if orbs is None:
            return "no legibles"
        return ", ".join(f"{name}={count}" for name, count in zip(CHARACTER_NAMES, orbs))

    # ── Punto de entrada del turno ──

    def execute_turn(self) -> None:
        """Ejecuta un turno completo del carrusel (llamado por el fighter con cartas en mano)."""
        wait_if_paused()

        # VERIFICACIÓN 1: estado del talento de DK
        screenshot, _ = capture_window()
        metrics = read_dk_talent_metrics(screenshot)

        # VERIFICACIÓN 2: orbes de cada personaje
        orbs = self._read_orbs_reliable()

        # VERIFICACIÓN 3: ultimate en la mano
        _ult_card, ult_char, ult_score = self._find_ult_in_hand()
        ult_text = f"{ult_char} ({ult_score:.2f})" if ult_char else "ninguna"

        print(
            f"— Verificaciones: talento={metrics['estado']} (brillo {metrics['brillo']:.0%}, "
            f"vivo {metrics['vivid_fraction']:.0%}) | orbes: {self._orbs_text(orbs)} | ulti en mano: {ult_text}"
        )

        stage = self._classify_stage(metrics, orbs, has_ult=ult_char is not None)

        if stage == "A" and orbs is not None:
            self._turn_a(orbs)
        elif stage == "B" and orbs is not None:
            self._turn_b(orbs)
        elif stage == "C" and orbs is not None:
            self._turn_c(orbs)
        else:
            self._filler_turn(orbs)

        self._ensure_turn_finished()

    # Requerido por IBattleStrategy; el carrusel no usa el protocolo de picks.
    def get_next_card_index(self, hand_of_cards: list[Card], picked_cards: list[Card], **kwargs) -> int:
        return SmarterBattleStrategy.get_next_card_index(hand_of_cards, picked_cards)
