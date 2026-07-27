"""Estrategia "carrusel" de la Estrategia Canopus (Demon King, 4 aliados).

Ciclo infinito de 3 turnos guiado por el estado del talento de DK. Solo se
mueven cartas para cargar orbes y se juegan las ultimates que van apareciendo.
Tristan (4to aliado, última posición a la derecha) queda fuera del carrusel:
el bot no mueve sus cartas para cargarle orbes ni lo usa para decidir turnos,
solo actúa sobre DK, Cusack y Galand. La excepción es la función "Seguro",
una secuencia completa de recuperación que corre sola cuando un turno A/B se
queda sin poder confirmar su ultimate:

  1. Queda armada (persistente, turno a turno) hasta que la ulti de Tristan
     esté en la mano.
  2. Ese turno: carga al personaje que quedó incompleto (o al de más orbes)
     hasta sus 5 orbes, lo VERIFICA, y recién entonces juega la ulti de
     Tristan como comodín (jugarla libera el lugar en la mano: sin eso,
     ninguna ulti de dk/cusack/galand puede aparecer jamás).
  3. Turnos siguientes: recarga a Tristan moviendo SOLO sus cartas
     (verificadas contra sus orbes) hasta confirmarlo de nuevo con 5. La
     ulti liberada por el comodín NO se juega en esta fase: queda guardada
     para el carrusel normal. Como el turno tiene 4 slots y la recarga
     necesita 5 movimientos, esto abarca más de un turno.
  4. Con Tristan verificado a 5, pasa lo que quede del turno moviendo SOLO
     cartas de Tristan (para no desacomodar el carrusel), desactiva el
     Seguro y la run continúa sola: el próximo Turno A/B del ciclo juega la
     ulti liberada que quedó esperando en la mano.

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

import time

import cv2
import numpy as np
import utilities.vision_images as vio
from utilities.app_config import wait_if_paused
from utilities.canopus_orb_reader import (
    CHARACTER_NAMES,
    read_ally_orbs,
    read_dk_talent_metrics,
    read_tristan_orbs,
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

# Templates de cartas por personaje (la ulti es siempre el último nombre).
# Tristan sí se identifica (necesario para detectar su ulti para la función
# "Seguro"), pero queda fuera de FOCUS_ORDER: el carrusel normal nunca lo
# elige como objetivo para mover cartas ni cargarle orbes.
CHAR_TEMPLATES = {
    "dk": ("dk_single", "dk_area", "dk_ulti"),
    "cusack": ("cusack_single", "cusack_orbe", "cusack_ulti"),
    "galand": ("galand_single", "galand_desventaja", "galand_ulti"),
    "tristan": ("tristan_single", "tristan_area", "tristan_ulti"),
}

# Nombre de template (string, el mismo que usan CHAR_TEMPLATES/ULT_TEMPLATES
# para comparar identidad) -> objeto Vision registrado en vision_images.py.
# Igual que el resto de las estrategias del bot (p. ej. deer_utilities.py con
# vio.thor_1, vio.jorm_ult...), en vez de cargar las imágenes a mano con un
# cv2.imread propio: así se comparte la carga perezosa, el aviso si falta el
# archivo, y el mismo lugar donde viven todas las demás imágenes del bot.
_TEMPLATE_IMAGES = {
    "dk_single": vio.canopus_dk_single,
    "dk_area": vio.canopus_dk_area,
    "dk_ulti": vio.canopus_dk_ulti,
    "cusack_single": vio.canopus_cusack_single,
    "cusack_orbe": vio.canopus_cusack_orbe,
    "cusack_ulti": vio.canopus_cusack_ulti,
    "galand_single": vio.canopus_galand_single,
    "galand_desventaja": vio.canopus_galand_desventaja,
    "galand_ulti": vio.canopus_galand_ulti,
    "tristan_single": vio.canopus_tristan_single,
    "tristan_area": vio.canopus_tristan_area,
    "tristan_ulti": vio.canopus_tristan_ulti,
}
ULT_TEMPLATES = {
    "dk": "dk_ulti",
    "cusack": "cusack_ulti",
    "galand": "galand_ulti",
    "tristan": "tristan_ulti",
}
# Prioridad en empates: el foco de la estrategia es DK, Cusack y Galand.
# Tristan queda fuera a propósito (ver CHAR_TEMPLATES).
FOCUS_ORDER = ("dk", "cusack", "galand")

_MIN_CARD_SCORE = 0.35  # score mínimo para asignar una carta a un personaje
_MIN_ULT_SCORE = 0.40  # score mínimo para dar por encontrada una ulti en mano
_ULT_VERIFY_SCORE = 0.32  # umbral (más bajo) para "la ulti sigue en mano" tras el click
_ULT_SCORE_DROP_MARGIN = 0.20  # caída de score post-click a partir de la cual asumimos que es otra carta
_MAX_ATTEMPTS = 5  # reintentos de un guion (con btn_reset entre intentos)
_MOVE_SLEEP = 1.2  # cooldown tras cada movimiento de carta (el juego necesita registrarlo)
_PRE_CLICK_SLEEP = 0.3  # cooldown antes de clickear la ulti (reintentos seguidos y rápidos)
_ULT_SLEEP = 0.6  # espera tras el doble click de la ultimate
_ULT_DOUBLE_CLICK_GAP = 0.08  # separación entre los dos clicks del doble click de la ulti
_TALENT_SLEEP = 2.5  # espera tras click al talento
_POST_TALENT_SLEEP = 1.0  # cooldown extra tras confirmar el talento, antes de empezar a mover cartas

# Punto del botón de talento (calibrado; escala estándar 550x940)
_TALENT_POINT = (275, 657)

# Frontera de vivid_fraction para arranque en frío: _1 (~4%) vs _2 (~17%)
_STAGE_1_VIVID_MAX = 0.10
_STAGE_2_VIVID_MAX = 0.24


def _load_template(name: str) -> np.ndarray | None:
    """Imagen cruda del template ``name`` vía ``_TEMPLATE_IMAGES``.

    ``Vision.needle_img`` ya cachea la carga (lazy, una sola vez por
    imagen) e imprime el aviso estándar del bot si el archivo falta, así
    que no hace falta un cv2.imread + lru_cache propios acá.
    """
    vision_image = _TEMPLATE_IMAGES.get(name)
    return vision_image.needle_img if vision_image is not None else None


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
        # True cuando algún turno A/B no pudo confirmar su ultimate. Queda
        # armado (persistente, turno a turno) hasta que _try_seguro logre
        # completar la fase del comodín: cargar al personaje pendiente a 5
        # orbes y confirmar la ulti de Tristan.
        self._pending_recovery: bool = False
        # Personaje cuyo turno quedó incompleto (el objetivo preferido de la
        # carga del Seguro). None => se elige al de más orbes.
        self._recovery_char: str | None = None
        # True tras jugar el comodín: fase de recarga de Tristan (ver
        # _continue_seguro_recharge). Mientras esté activa, cada turno se
        # dedica a jugar la ulti liberada y recargar a Tristan, hasta
        # verificarlo de nuevo con 5 orbes.
        self._seguro_recharging: bool = False
        # True cuando el turno actual quedó con slots sin llenar porque no
        # hubo un movimiento seguro para completarlos (ver _ensure_turn_finished).
        # El fighter puede volver a llamar a execute_turn() creyendo que es un
        # turno nuevo cuando en realidad seguimos atascados en el mismo; en
        # ese caso hay que evitar repetir todo el guion (reclickear el
        # talento, replanear A/B/C) y solo reintentar completar los slots.
        self._turn_pending_fill = False

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

        Solo considera personajes de FOCUS_ORDER: la ulti de Tristan NUNCA se
        juega por este camino (turno normal), solo mediante la función
        "Seguro" (``_try_seguro``), que la busca explícitamente.
        """
        best_card, best_char, best_score = None, None, 0.0
        for _idx, card, char, tpl, score in self._hand_snapshot():
            if char is None or char not in FOCUS_ORDER or tpl != ULT_TEMPLATES[char]:
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
        template = vio.canopus_btn_reset.needle_img
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
            return False
        center = (loc[0] + shape[1] // 2, loc[1] + shape[0] // 2)
        click_im(center, window_location)
        time.sleep(1.2)
        return True

    def _pick_move_target(self, snapshot, origin_pos: int) -> Card | None:
        """Carta ocupada más cercana al origen para soltar la carta que se mueve.

        Evita dos cosas, en orden de prioridad:
        1. La ulti lista de CUALQUIER personaje (propio o no). Soltar una
           carta encima de una ultimate lista puede jugarla/gastarla sin
           querer, fuera del control de ``_play_specific_ult`` (visto en
           vivo: DK tenía 5 orbes y su ulti en mano, y quedó en 0 orbes en
           medio de un Turno C que ni siquiera juega ultimates).
        2. Si puede, evita también un posible merge (mismo personaje y rango
           que la carta que se mueve): es reversible, así que se acepta ese
           riesgo antes que quedarse sin mover, pero la ulti lista NUNCA se
           usa como objetivo, pase lo que pase.
        """
        origin_idx, origin_card, origin_char, _tpl, _s = snapshot[origin_pos]
        candidates = sorted(
            (item for pos, item in enumerate(snapshot) if pos != origin_pos),
            key=lambda item: abs(item[0] - origin_idx),
        )
        non_ult = [
            item
            for item in candidates
            if not (item[2] is not None and item[3] == ULT_TEMPLATES.get(item[2]))
        ]
        for _idx, card, char, _tpl2, _s2 in non_ult:
            if char == origin_char and card.card_rank == origin_card.card_rank:
                # Posible merge (mismo personaje y rango): probar otro objetivo
                continue
            return card
        # Sin objetivo "seguro" que evite un merge: usar el más cercano que no
        # sea una ulti lista, si hay alguno.
        return non_ult[0][1] if non_ult else None

    def _move_char_cards(self, char: str, times: int) -> int:
        """Mueve hasta ``times`` cartas del personaje dado, verificando cada una.

        Tras cada movimiento comprueba que los orbes de ``char`` subieron en 1
        (tope 5). Si un movimiento no se puede verificar, corta el bloque de
        inmediato en vez de seguir moviendo cartas a ciegas: el llamador
        (con su propio bucle de reintentos) es quien decide resetear y volver
        a intentar. Devuelve cuántos movimientos quedaron verificados.

        Antes de cada movimiento comprueba que sigan quedando slots vacíos
        (``_empty_slots``, la misma señal de "es mi turno" que usa el resto
        del bot -ver ``CanopusFighter.count_empty_card_slots``-): si el
        turno ya terminó (o pasó al enemigo) arrastrar una carta y leer
        orbes no tiene sentido -las lecturas en ese momento pueden salir
        con cualquier cosa, como se vio en vivo (los 3 aliados cayendo a
        casi cero de golpe, algo que un movimiento real nunca produce)-, así
        que hay que cortar ahí en vez de seguir actuando a ciegas.
        """
        idx = CHARACTER_NAMES.index(char)
        current_orbs = self._read_orbs_reliable()
        if current_orbs is None:
            print(f"No puedo leer los orbes antes de mover cartas de {char}; no arriesgo el movimiento.")
            return 0
        moved = 0
        for _ in range(times):
            wait_if_paused()
            if self._empty_slots() <= 0:
                print(f"Sin slots vacíos; el turno ya terminó, corto el movimiento de {char}.")
                break
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
            drag_im(origin_point, target_point, window_location, sleep_after_click=0.1, drag_duration=0.25)
            time.sleep(_MOVE_SLEEP)

            expected = min(5, current_orbs[idx] + 1)
            new_orbs = self._read_orbs_reliable()
            if new_orbs is None or new_orbs[idx] < expected:
                print(
                    f"Movimiento de {char} no verificado (orbes antes={current_orbs}, "
                    f"después={new_orbs}); corto el bloque para resetear."
                )
                break
            current_orbs = new_orbs
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

    def _play_specific_ult(self, char: str, orig_score: float) -> bool:
        """Juega la carta de ulti de ``char``, verificando el click.

        Re-ubica la carta (posición fresca, vía ``_char_ult_in_hand``) recién
        después de la espera previa al click, nunca antes: identificar cartas
        es lento (varios ``cv2.matchTemplate`` por carta) y en una laptop
        lenta ese tiempo alcanza para que la mano se reacomode, dejando
        desactualizada una posición calculada de antemano y haciendo que el
        click caiga en otra carta. Por eso no se reutiliza el ``Card`` de
        ``_find_ult_in_hand``/la iteración anterior: siempre se vuelve a
        buscar la carta justo antes de cada click.

        Si la carta ya no está, o su score cayó demasiado respecto al
        original (una carta nueva puede matchear el mismo template por
        casualidad, con un score mucho más bajo — visto en vivo: 0.87 -> 0.58
        -> 0.36), se da por jugada la ulti en vez de seguir clickeando algo
        que ya no es la misma carta.

        Cada intento hace DOBLE click (un solo click a veces no alcanza a
        registrarse) y los reintentos van seguidos y rápidos (``_PRE_CLICK_SLEEP``
        y ``_ULT_SLEEP`` cortos): mejor varios intentos rápidos sobre la
        posición recién verificada que esperar de más entre cada uno.
        """
        for attempt in range(1, _MAX_ATTEMPTS + 1):
            time.sleep(_PRE_CLICK_SLEEP)

            card, score = self._char_ult_in_hand(char)
            if card is None:
                print(f"Verificado: la ultimate de {char} ya no está en la mano.")
                return True
            if score < orig_score - _ULT_SCORE_DROP_MARGIN:
                print(
                    f"El match de la ultimate de {char} cayó de {orig_score:.2f} a {score:.2f}; "
                    "asumo que ya se jugó y no clickeo otra carta."
                )
                return True

            _, window_location = capture_window()
            click_im(card.rectangle, window_location)
            time.sleep(_ULT_DOUBLE_CLICK_GAP)
            click_im(card.rectangle, window_location)
            time.sleep(_ULT_SLEEP)

        print(f"No pude confirmar que la ultimate de {char} se jugara tras {_MAX_ATTEMPTS} clicks.")
        return False

    # ── Función "Seguro" ──

    def _try_seguro(self) -> bool:
        """Red de contención: usa la ulti de Tristan como comodín si el turno
        anterior no se pudo confirmar la ultimate correcta.

        Fase del comodín. Queda ARMADA de forma persistente (no se descarta
        si un turno no se puede cumplir): cada turno nuevo se reintenta
        hasta lograr la secuencia completa, en este orden:

        1. La ulti de Tristan tiene que estar en la mano; si no está, el
           turno sigue su curso normal y se reintenta el próximo.
        2. Carga al personaje que quedó incompleto (``_recovery_char``, o el
           de más orbes) hasta sus 5 orbes y lo VERIFICA (ver
           ``_charge_recovery_char_to_full``). Su ulti NO va a aparecer en
           la mano todavía -la mano está al tope de cartas hasta que se
           juegue/descarte alguna-, así que acá solo se carga.
        3. Solo con la carga verificada juega la ulti de Tristan: al
           jugarla libera el lugar en la mano, y la ulti recién cargada va
           a poder aparecer el turno siguiente.
        4. Confirmado el comodín, pasa a la fase de recarga
           (``_seguro_recharging``, ver ``_continue_seguro_recharge``) y
           aprovecha los slots que queden de este turno para arrancarla.

        Si la carga o el click del comodín no se pueden confirmar, el
        Seguro SIGUE armado y lo reintenta el próximo turno (devuelve True
        igual: el turno ya se gastó en los intentos).
        """
        if not self._pending_recovery:
            return False

        tristan_card, tristan_score = self._char_ult_in_hand("tristan")
        if tristan_card is None:
            print("[Seguro] Armado, pero la ulti de Tristan no está en la mano; sigo el turno normal y reintento después.")
            return False

        print(f"[Seguro] Ulti de Tristan en mano (score {tristan_score:.2f}); inicio la recuperación.")

        if not self._charge_recovery_char_to_full():
            print("[Seguro] Carga a 5 orbes sin verificar; no gasto la ulti de Tristan y reintento el próximo turno.")
            return True

        if not self._play_specific_ult("tristan", tristan_score):
            print("[Seguro] No pude confirmar la ulti de Tristan; reintento el próximo turno.")
            return True

        # Al jugar su ulti, los orbes propios de Tristan bajan a 0. Desde acá
        # la fase de recarga se encarga de reconstruirle la barra turno a
        # turno hasta verificarlo de nuevo con 5 (sin eso, el Seguro no
        # tendría comodín para la próxima falla).
        print("[Seguro] Comodín jugado; ahora toca recargar a Tristan hasta sus 5 orbes.")
        self._pending_recovery = False
        self._recovery_char = None
        self._seguro_recharging = True
        # Aprovechar los slots que queden de este turno para arrancar la recarga.
        self._recharge_tristan()
        return True

    def _recharge_tristan(self) -> int:
        """Mueve hasta 5 cartas de Tristan para reconstruir su barra de orbes
        tras usar su ulti en el Seguro, verificando cada movimiento contra
        ``read_tristan_orbs`` (igual de estricto que ``_move_char_cards``,
        pero contra su lectura separada en vez de la de los 3 aliados del
        carrusel). Si el hueco de cartas de Tristan en la mano se agota antes
        de las 5, el resto queda para una futura recarga.

        Antes de cada movimiento comprueba que sigan quedando slots vacíos
        (``_empty_slots``, la misma señal de "es mi turno" que usa el resto
        del bot): un turno solo tiene 4, así que puede que no alcance para
        las 5 cartas y hay que cortar ahí en vez de seguir arrastrando cartas
        sin que el juego registre nada.

        Su propia carta de ulti nunca se elige como origen del movimiento
        (se excluye, no solo se deprioriza): arrastrarla como si fuera una
        carta normal no la mueve -el juego no la trata como una carta de
        fusión más- y deja la recarga girando en falso sobre la misma carta.
        """
        screenshot, _ = capture_window()
        current = read_tristan_orbs(screenshot)
        if current is None:
            print("[Seguro] No puedo leer los orbes de Tristan para recargarlo; no arriesgo el movimiento.")
            return 0

        moved = 0
        for _ in range(5):
            if current >= 5:
                break
            wait_if_paused()
            if self._empty_slots() <= 0:
                print("[Seguro] Sin slots vacíos; corto la recarga de Tristan por este turno.")
                break
            snapshot = self._hand_snapshot()
            # Su carta de ulti se EXCLUYE como origen (arrastrarla no la
            # mueve y deja la recarga girando en falso), pero verla en la
            # mano NO da la recarga por completa: la única señal válida de
            # barra llena son los 5 orbes leídos, no un match de template
            # (puede ser un falso positivo).
            own = [
                (pos, item)
                for pos, item in enumerate(snapshot)
                if item[2] == "tristan" and item[3] != ULT_TEMPLATES["tristan"]
            ]
            if not own:
                print("[Seguro] No encuentro cartas de Tristan en la mano para recargarlo.")
                break
            own.sort(key=lambda pair: -pair[1][4])
            origin_pos, (origin_idx, origin_card, _c, _t, score) = own[0]
            target_card = self._pick_move_target(snapshot, origin_pos)
            if target_card is None:
                print("[Seguro] No hay carta objetivo para recargar a Tristan.")
                break
            _, window_location = capture_window()
            origin_point = get_click_point_from_rectangle(origin_card.rectangle)
            target_point = get_click_point_from_rectangle(target_card.rectangle)
            drag_im(origin_point, target_point, window_location, sleep_after_click=0.1, drag_duration=0.25)
            time.sleep(_MOVE_SLEEP)

            expected = min(5, current + 1)
            screenshot, _ = capture_window()
            new_orbs = read_tristan_orbs(screenshot)
            if new_orbs is None or new_orbs < expected:
                print(
                    f"[Seguro] Movimiento de recarga de Tristan no verificado (antes={current}, "
                    f"después={new_orbs}); corto la recarga."
                )
                break
            current = new_orbs
            moved += 1
        return moved

    def _continue_seguro_recharge(self) -> bool:
        """Fase de recarga del Seguro, turno a turno, tras jugar el comodín.

        Mientras ``_seguro_recharging`` esté activa, cada turno se dedica
        EXCLUSIVAMENTE a mover cartas de Tristan (verificadas contra sus
        orbes) hasta confirmarlo de nuevo con 5. Como el turno tiene 4
        slots y la recarga necesita 5 movimientos, esto normalmente abarca
        más de un turno.

        La ulti liberada por el comodín (dk/cusack/galand) NO se juega en
        esta fase, aunque ya aparezca en la mano: queda guardada, y la va a
        jugar el carrusel normal (Turno A/B) cuando el Seguro se desactive
        y la run retome su ciclo.

        Cuando Tristan queda VERIFICADO con sus 5 orbes: pasar lo que quede
        del turno moviendo SOLO cartas de Tristan (ver
        ``_pass_turn_with_tristan``) y desactivar el Seguro.
        """
        if not self._seguro_recharging:
            return False

        print("[Seguro] Fase de recarga: este turno solo muevo cartas de Tristan.")
        self._recharge_tristan()

        # La ÚNICA señal que cierra el Seguro son los 5 orbes de Tristan
        # leídos de su barra. Ver su carta de ulti en la mano NO cuenta
        # (un match de template puede ser un falso positivo y cerraría la
        # recarga antes de tiempo, dejando al Seguro sin comodín real).
        screenshot, _ = capture_window()
        tristan_orbs = read_tristan_orbs(screenshot)
        tristan_full = tristan_orbs is not None and tristan_orbs >= 5
        if tristan_full:
            print("[Seguro] Verificado: Tristan volvió a sus 5 orbes. Paso el turno con sus cartas y desactivo el Seguro.")
            self._seguro_recharging = False
        else:
            print(f"[Seguro] Tristan sigue con {tristan_orbs} orbe(s); continúo la recarga el próximo turno.")
        # En ambos casos, los slots que queden de este turno se consumen con
        # cartas de Tristan (nunca de dk/cusack/galand): lo hace el llamador
        # (execute_turn) vía _pass_turn_with_tristan.
        return True

    def _pass_turn_with_tristan(self) -> None:
        """Consume los slots restantes del turno moviendo SOLO cartas de Tristan.

        Es el relleno de turno de TODA la fase de recarga (lo usa
        ``execute_turn`` en lugar de ``_ensure_turn_finished``, que movería
        cartas de dk/cusack/galand y desacomodaría los conteos del
        carrusel). Si Tristan sigue por debajo de 5, estos movimientos
        además le suman orbes (aunque acá no se verifiquen: la próxima
        pasada de ``_continue_seguro_recharge`` los relee en frío); si ya
        está al tope, solo gastan los slots para que el turno termine. El
        avance se mide por la fila de slots vacíos, re-chequeada antes de
        cada movimiento. Solo si se acaban las cartas de Tristan en la mano
        cae al relleno normal, para no dejar el turno colgado.
        """
        for _ in range(6):
            wait_if_paused()
            if self._empty_slots() <= 0:
                return
            snapshot = self._hand_snapshot()
            own = [
                (pos, item)
                for pos, item in enumerate(snapshot)
                if item[2] == "tristan" and item[3] != ULT_TEMPLATES["tristan"]
            ]
            if not own:
                print("[Seguro] Sin cartas de Tristan para pasar el turno; uso el relleno normal.")
                self._ensure_turn_finished()
                return
            own.sort(key=lambda pair: -pair[1][4])
            origin_pos, (_origin_idx, origin_card, _c, _t, _score) = own[0]
            target_card = self._pick_move_target(snapshot, origin_pos)
            if target_card is None:
                self._ensure_turn_finished()
                return
            _, window_location = capture_window()
            drag_im(
                get_click_point_from_rectangle(origin_card.rectangle),
                get_click_point_from_rectangle(target_card.rectangle),
                window_location,
                sleep_after_click=0.1,
                drag_duration=0.25,
            )
            time.sleep(_MOVE_SLEEP)

    def _char_by_orbs(
        self, orbs: list[int], predicate, label: str, exclude: int | None = None
    ) -> int | None:
        """Índice del personaje cuyo conteo cumple ``predicate``; empates por FOCUS_ORDER.

        ``exclude`` descarta un índice ya asignado a otro rol del mismo turno,
        para no devolver el mismo personaje dos veces (p. ej. en Turno B, que
        necesita un personaje con 3 orbes y OTRO con 0).
        """
        matching = [
            i for i, count in enumerate(orbs) if predicate(count) and i != exclude
        ]
        if not matching:
            return None
        matching.sort(key=lambda i: FOCUS_ORDER.index(CHARACTER_NAMES[i]))
        return matching[0]

    def _charge_recovery_char_to_full(self) -> bool:
        """Carga hasta 5 orbes (VERIFICADOS) al personaje objetivo del Seguro.

        Preferencia: ``_recovery_char`` (el que quedó con sus movimientos
        incompletos en el turno que falló); si ya está a 5 o no se guardó,
        el personaje de FOCUS_ORDER con más orbes por debajo de 5. Si los
        tres ya están a 5, no hay nada que cargar y la carga cuenta como
        verificada.

        Ojo: su ultimate NO va a aparecer en la mano por más que llegue a 5
        -la mano está al tope de cartas, y una nueva carta (incluida una
        ulti recién habilitada) no entra hasta que se juegue/descarte
        alguna-. Por eso acá solo se carga: jugar la ulti de Tristan justo
        después (ver _try_seguro) es lo que libera el lugar para que esta
        ulti aparezca recién el turno siguiente.
        """
        current = self._read_orbs_reliable()
        if current is None:
            print("[Seguro] No puedo leer los orbes para la carga; no arriesgo movimientos.")
            return False

        target = None
        if self._recovery_char in CHARACTER_NAMES:
            idx = CHARACTER_NAMES.index(self._recovery_char)
            if current[idx] < 5:
                target = idx
        if target is None:
            below = [i for i in range(len(current)) if current[i] < 5]
            if not below:
                print("[Seguro] Los tres personajes ya están a 5 orbes; nada que cargar.")
                return True
            target = max(below, key=lambda i: (current[i], -FOCUS_ORDER.index(CHARACTER_NAMES[i])))

        char = CHARACTER_NAMES[target]
        before = current[target]
        needed = 5 - before
        print(f"[Seguro] Cargo a {char} ({before} orbes) hasta 5 antes de jugar la ulti de Tristan.")
        moved = self._move_char_cards(char, needed)

        after = self._read_orbs_reliable()
        if moved == needed and after is not None and after[target] >= 5:
            print(f"[Seguro] Verificado: {char} quedó con sus 5 orbes.")
            return True
        got = "?" if after is None else after[target]
        print(f"[Seguro] Carga de {char} sin verificar (movidas {moved}/{needed}, quedó con {got}).")
        return False

    # ── Guiones de turno ──

    def _play_any_ready_ult(self) -> bool:
        """Juega la mejor ultimate lista en la mano, sea de quien sea.

        Ninguna ulti de dk/cusack/galand puede aparecer en la mano sin que
        antes Tristan libere un lugar jugando la suya (ver _try_seguro): la
        mano queda al tope de cartas y no entra ninguna nueva -ulti recién
        habilitada incluida- hasta que se juegue/descarte alguna. Por eso la
        que aparezca lista puede ser de un personaje distinto al que este
        turno cargó, y hay que jugarla igual: es la única forma en que el
        carrusel avanza.

        Devuelve True solo si una ultimate quedó confirmada como jugada. Un
        turno A/B que termina sin eso (no había ninguna, o el click no se
        pudo confirmar) es la señal de que el ciclo se quedó sin ultimates
        y el llamador debe armar el Seguro.
        """
        ult_card, ult_char, ult_score = self._find_ult_in_hand()
        if ult_card is None:
            print("[Turno] Sin ninguna ultimate lista en la mano.")
            return False
        if not self._play_specific_ult(ult_char, ult_score):
            print(f"[Turno] Encontré la ulti de {ult_char} pero no pude confirmar que se jugó.")
            return False
        return True

    def _turn_a(self, orbs: list[int]) -> None:
        print(f"[Turno A] Talento ACTIVO. Orbes: {self._orbs_text(orbs)}")

        # Chequeo de Tristan: se hace SIEMPRE al entrar a Turno A (antes de
        # clickear el talento). Si le quedan 0 o 1 orbes, es que se usó su
        # ulti hace poco y todavía no se lo recargó -sin esto, el Seguro se
        # queda sin comodín para la próxima falla-. Nos enfocamos en
        # recargarlo y NO clickeamos el talento ni seguimos con el resto del
        # guion este turno: como el talento sigue "activo", el próximo turno
        # vuelve a entrar acá y revisa de nuevo, hasta que ya no haga falta.
        screenshot, _ = capture_window()
        tristan_orbs = read_tristan_orbs(screenshot)
        if tristan_orbs is not None and tristan_orbs <= 1:
            print(f"[Turno A] Tristan con {tristan_orbs} orbe(s); me enfoco en recargarlo antes de seguir.")
            recharged = self._recharge_tristan()
            print(f"[Turno A] Recarga de Tristan: {recharged} movimiento(s) verificado(s).")
            # Lo que quede del turno también se pasa SOLO con cartas de
            # Tristan, para no desacomodar el carrusel con movimientos de
            # otros personajes.
            self._pass_turn_with_tristan()
            return

        self._click_talent()

        current = orbs
        char = None
        for attempt in range(1, _MAX_ATTEMPTS + 1):
            if self._empty_slots() <= 0:
                print("[Turno A] Sin slots vacíos; el turno ya terminó, no reintento más.")
                break
            target = min(range(len(current)), key=lambda i: (current[i], FOCUS_ORDER.index(CHARACTER_NAMES[i])))
            char = CHARACTER_NAMES[target]
            before = current[target]
            expected = min(5, before + 3)
            print(f"[Turno A] Moviendo 3 cartas de {char} ({before} orbes; espero {expected}).")
            moved = self._move_char_cards(char, 3)

            after = self._read_orbs_reliable()
            if moved == 3 and after is not None and after[target] >= expected:
                print(f"[Turno A] Verificado: {char} quedó con {after[target]} orbes.")
                if not self._play_any_ready_ult():
                    print("[Turno A] Sin ultimate confirmada este turno; armo el Seguro.")
                    self._pending_recovery = True
                    self._recovery_char = char
                self._next_stage = "B"
                return
            else:
                got = "?" if after is None else after[target]
                print(f"[Turno A] Verificación fallida (movidas {moved}, {char} con {got} orbes).")

            if attempt < _MAX_ATTEMPTS:
                self._press_reset()
                current = self._read_orbs_reliable() or current

        print("[Turno A] Agotados los intentos; termino el turno como pueda y armo el Seguro.")
        self._pending_recovery = True
        self._recovery_char = char
        self._next_stage = "B"

    def _turn_b(self, orbs: list[int]) -> None:
        print(f"[Turno B] Talento en cooldown (etapa 1). Orbes: {self._orbs_text(orbs)}")

        current = orbs
        char3 = char0 = None
        for attempt in range(1, _MAX_ATTEMPTS + 1):
            if self._empty_slots() <= 0:
                print("[Turno B] Sin slots vacíos; el turno ya terminó, no reintento más.")
                break
            idx3 = self._char_by_orbs(current, lambda n: n == 3, "tener 3 orbes")
            if idx3 is None:
                idx3 = self._char_by_orbs(current, lambda n: n < 5, "tener menos de 5 orbes (fallback)")

            # idx0 siempre excluye a idx3: nunca debe ser el mismo personaje para los dos roles.
            idx0 = self._char_by_orbs(current, lambda n: n == 0, "tener 0 orbes", exclude=idx3)
            if idx0 is None:
                idx0 = self._char_by_orbs(
                    current,
                    lambda n: n == min(c for i, c in enumerate(current) if i != idx3),
                    "tener los menos orbes (fallback)",
                    exclude=idx3,
                )
            if idx3 is None or idx0 is None:
                # Nadie a quien cargarle más orbes (p. ej. los tres ya al
                # tope de 5): no hay nada más que mover, pero igual puede
                # haber una ulti lista para jugar (la que se haya liberado
                # con la última ulti jugada), así que la revisamos antes de
                # dar el turno por perdido.
                print("[Turno B] Nadie para cargar (¿orbes al tope?); reviso si hay alguna ulti lista.")
                if not self._play_any_ready_ult():
                    print("[Turno B] Tampoco hay ulti lista: el ciclo quedó trabado; armo el Seguro.")
                    self._pending_recovery = True
                    self._recovery_char = None
                self._next_stage = "C"
                return

            char3, char0 = CHARACTER_NAMES[idx3], CHARACTER_NAMES[idx0]
            before3, before0 = current[idx3], current[idx0]
            print(f"[Turno B] 2 movimientos de {char3} ({before3} orbes) y 1 de {char0} ({before0} orbes).")
            moved3 = self._move_char_cards(char3, 2)
            # Si el bloque de char3 ya falló, no arriesgar el de char0: de todas formas el reset lo deshace.
            moved0 = self._move_char_cards(char0, 1) if moved3 == 2 else 0

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
                if not self._play_any_ready_ult():
                    print("[Turno B] Sin ultimate confirmada este turno; armo el Seguro.")
                    self._pending_recovery = True
                    self._recovery_char = char3
                self._next_stage = "C"
                return
            else:
                print(f"[Turno B] Verificación fallida (movidas {moved3}+{moved0}, orbes {after}).")

            if attempt < _MAX_ATTEMPTS:
                self._press_reset()
                current = self._read_orbs_reliable() or current

        print("[Turno B] Agotados los intentos; termino el turno como pueda.")
        if char3 is not None and char0 is not None:
            print("[Turno B] Armo el Seguro para recuperar el ciclo.")
            self._pending_recovery = True
            self._recovery_char = char3
        self._next_stage = "C"

    def _turn_c(self, orbs: list[int]) -> None:
        print(f"[Turno C] Talento en cooldown (etapa 2). Orbes: {self._orbs_text(orbs)} — sin ultimate.")

        current = orbs
        for attempt in range(1, _MAX_ATTEMPTS + 1):
            if self._empty_slots() <= 0:
                print("[Turno C] Sin slots vacíos; el turno ya terminó, no reintento más.")
                break
            idx1 = self._char_by_orbs(current, lambda n: n == 1, "tener 1 orbe")
            if idx1 is None:
                idx1 = self._char_by_orbs(current, lambda n: n == min(current), "tener los menos orbes (fallback)")
            if idx1 is None:
                print("[Turno C] No encuentro personaje para mover.")
                break

            char = CHARACTER_NAMES[idx1]
            print(f"[Turno C] Moviendo 4 cartas de {char} ({current[idx1]} orbes).")
            moved = self._move_char_cards(char, 4)
            if moved == 4:
                print(f"[Turno C] Verificado: 4 movimientos de {char} completados.")
                self._next_stage = None
                return
            print(f"[Turno C] Verificación fallida (movidas {moved}/4 de {char}).")

            if attempt < _MAX_ATTEMPTS:
                self._press_reset()
                current = self._read_orbs_reliable() or current

        print("[Turno C] Agotados los intentos; termino el turno como pueda.")
        # Cierra el ciclo: el siguiente turno debería volver a tener el talento ACTIVO.
        self._next_stage = None

    def _filler_turn(self, orbs: list[int] | None) -> None:
        """Sin señal clara del ciclo: mover cartas del personaje con menos orbes para no estancarse."""
        print("[Turno ?] Sin señal clara del ciclo; muevo cartas del personaje con menos orbes.")
        current = orbs if orbs is not None else [0] * len(CHARACTER_NAMES)
        for attempt in range(1, _MAX_ATTEMPTS + 1):
            if self._empty_slots() <= 0:
                print("[Turno ?] Sin slots vacíos; el turno ya terminó, no reintento más.")
                return
            target = min(range(len(current)), key=lambda i: (current[i], FOCUS_ORDER.index(CHARACTER_NAMES[i])))
            char = CHARACTER_NAMES[target]
            moved = self._move_char_cards(char, 2)
            if moved == 2:
                return
            print(f"[Turno ?] Verificación fallida (movidas {moved}/2 de {char}).")
            if attempt < _MAX_ATTEMPTS:
                self._press_reset()
                current = self._read_orbs_reliable() or current

        print("[Turno ?] Agotados los intentos; sigo sin poder mover cartas de forma segura.")

    def _ensure_turn_finished(self) -> None:
        """Rellenar los slots que falten con movimientos del personaje con menos orbes.

        Cada movimiento se verifica (ver ``_move_char_cards``); si uno falla,
        resetea y reintenta en la siguiente ronda en vez de arriesgar una
        jugada al azar. Si tras ``_MAX_ATTEMPTS`` rondas seguidas fallidas
        seguimos sin poder mover nada, se deja el turno como esté.
        """
        failed_in_a_row = 0
        for _ in range(6):
            wait_if_paused()
            if self._empty_slots() == 0:
                return
            orbs = self._read_orbs_reliable() or [0] * len(CHARACTER_NAMES)
            target = min(range(len(orbs)), key=lambda i: (orbs[i], FOCUS_ORDER.index(CHARACTER_NAMES[i])))
            if self._move_char_cards(CHARACTER_NAMES[target], 1) == 0:
                failed_in_a_row += 1
                if failed_in_a_row >= _MAX_ATTEMPTS:
                    print("No pude completar el turno tras varios intentos; no arriesgo un movimiento al azar.")
                    return
                print("Movimiento de relleno no verificado; reseteo y reintento.")
                self._press_reset()
            else:
                failed_in_a_row = 0
            time.sleep(0.4)

    def _dispatch_cycle_turn(self, metrics: dict, orbs: list[int] | None, ult_char: str | None) -> None:
        """Corre el guion normal del ciclo (A/B/C o filler) según ``metrics``/
        ``orbs``/``ult_char`` ya leídos. El llamador decide si reusar
        verificaciones ya hechas este turno o volver a leerlas (p. ej. tras
        la recarga de Tristan, que sí cambia el estado real del juego).
        """
        stage = self._classify_stage(metrics, orbs, has_ult=ult_char is not None)

        if stage == "A" and orbs is not None:
            self._turn_a(orbs)
        elif stage == "B" and orbs is not None:
            self._turn_b(orbs)
        elif stage == "C" and orbs is not None:
            self._turn_c(orbs)
        else:
            self._filler_turn(orbs)

    @staticmethod
    def _orbs_text(orbs: list[int] | None) -> str:
        if orbs is None:
            return "no legibles"
        return ", ".join(f"{name}={count}" for name, count in zip(CHARACTER_NAMES, orbs))

    # ── Punto de entrada del turno ──

    def execute_turn(self) -> None:
        """Ejecuta un turno completo del carrusel (llamado por el fighter con cartas en mano)."""
        wait_if_paused()

        if self._turn_pending_fill:
            # No es un turno nuevo: el anterior quedó con slots sin llenar (a
            # propósito, para no arriesgar un movimiento) y el fighter volvió
            # a llamar acá porque todavía ve slots vacíos. Repetir todo el
            # guion (reclickear el talento, replanear A/B/C) sería repetir
            # trabajo ya hecho y arriesgar acciones de más; solo reintentamos
            # completar lo que falte.
            print("Turno todavía sin completar; reintento solo llenar los slots que faltan.")
            self._ensure_turn_finished()
            self._turn_pending_fill = self._empty_slots() > 0
            return

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

        # Fase 1 del Seguro (armado): en cuanto la ulti de Tristan esté en la
        # mano, carga al personaje pendiente a 5 orbes y juega el comodín.
        # Si se dispara, reemplaza al guion normal A/B/C de este turno.
        if self._try_seguro():
            if self._seguro_recharging:
                # Comodín ya jugado: lo que quede del turno es recarga, así
                # que se rellena SOLO con cartas de Tristan.
                self._pass_turn_with_tristan()
            else:
                self._ensure_turn_finished()
            self._turn_pending_fill = self._empty_slots() > 0
            return

        # Fase 2 del Seguro (tras el comodín): recargar a Tristan turno a
        # turno hasta verificarlo con 5 orbes; recién ahí se desactiva y la
        # run sigue sola. El relleno del turno también es SOLO con cartas de
        # Tristan: usar el relleno normal acá movería cartas de
        # dk/cusack/galand al azar y desacomodaría el carrusel.
        if self._continue_seguro_recharge():
            self._pass_turn_with_tristan()
            self._turn_pending_fill = self._empty_slots() > 0
            return

        self._dispatch_cycle_turn(metrics, orbs, ult_char)

        self._ensure_turn_finished()
        self._turn_pending_fill = self._empty_slots() > 0

    # Requerido por IBattleStrategy; el carrusel no usa el protocolo de picks.
    def get_next_card_index(self, hand_of_cards: list[Card], picked_cards: list[Card], **kwargs) -> int:
        return SmarterBattleStrategy.get_next_card_index(hand_of_cards, picked_cards)
