"""Lector de orbes de ultimate para la Estrategia Canopus (formación de 3 aliados).

Cada aliado tiene sobre su cabeza una barra de vida verde y debajo una fila de
5 orbes que se llenan de dorado al mover sus cartas. Este módulo localiza las
barras HP por color (la cámara del juego se desplaza, así que no sirven
coordenadas fijas) y cuenta los orbes llenos de cada personaje.

La formación de los 3 aliados es rígida: las distancias entre barras son
constantes aunque la cámara se mueva. Cada barra candidata "vota" por dónde
estaría el origen de la formación completa; gana la hipótesis con más barras
de apoyo y las barras tapadas (números de daño, efectos, barra vacía por daño)
se reconstruyen a partir de ese origen. Basta con ver 2 de las 3 barras.

Orden en pantalla (izquierda a derecha): DK, Cusack, Galand. Tristan (4ta
posición, a la derecha) queda fuera de la estrategia: el bot no mueve sus
cartas ni intenta leer sus orbes.

Calibrado con captura real de la pelea (2026-07-26), a escala nativa de la
ventana estándar del bot (captura de 550x940):
- Barra HP llena: franja verde-lima de ~77x7 px, en y ∈ [450, 505] aprox.
- Offsets respecto a la barra izquierda (slot 0): (+129, +20), (+272, -19).
  Formación escalonada (Galand, slot 2, es el elevado).
- Orbes: fila de 5, empieza en (barra_x+2, barra_y+alto+8), paso 14.0 px,
  celda de 13x13 px.
- Orbe lleno: S media 130-141, V media 183-216. Vacío: S 69-116, V 77-134.

Las capturas de otro tamaño se normalizan automáticamente a 550x940 antes de
medir, así que ``read_ally_orbs`` funciona aunque la ventana no esté al tamaño
estándar.

Para recalibrar (imprime candidatas y offsets, y guarda una imagen anotada):

    python utilities/canopus_orb_reader.py captura.png
    python utilities/canopus_orb_reader.py            # captura la ventana 7DS en vivo
"""

import cv2
import numpy as np

# Orden de los personajes en pantalla (izquierda a derecha). Actualizado
# 2026-07-26: el foco de la estrategia es DK, Cusack y Galand; Tristan (4ta
# posición, a la derecha) no se detecta ni se juega.
ORB_DK = 0
ORB_CUSACK = 1
ORB_GALAND = 2
CHARACTER_NAMES = ("dk", "cusack", "galand")
NUM_ALLIES = 3

# Región donde viven las barras HP de los aliados
_BAND_Y_MIN = 400
_BAND_Y_MAX = 650
_BAND_X_MAX = 540  # a la derecha solo hay iconos de UI (grises, no disparan el filtro verde)

# Geometría barra -> orbes (idéntica a la formación de 3; no depende de ella)
_ORB_DX = 2
_ORB_DY = 8  # desde el borde INFERIOR de la barra verde
_ORB_PITCH = 14.0
_ORB_SIZE = 13
_ORBS_PER_ALLY = 5

# Offsets rígidos de la formación de 3, respecto a la barra de más a la
# izquierda (slot 0). Recalibrados 2026-07-26 tras reordenar el equipo a
# DK, Cusack, Galand (captura nativa 550x940). Tristan (4ta posición) queda
# fuera de la formación: no se detecta.
_FORMATION = (
    (0, 0),
    (129, 20),
    (272, -19),
)
_TOL_X = 28
_TOL_Y = 13

# Mínimo de barras reales visibles para fiarnos de una hipótesis de formación
_MIN_BARS_SUPPORT = 2
# Si dos orígenes distintos empatan en apoyo y su desviación difiere menos que
# esto, la lectura es ambigua (p. ej. un par de barras que encaja en dos
# posiciones de la formación) y preferimos devolver None a equivocar el slot.
_AMBIGUITY_DEV_MARGIN = 6

# Umbrales de color (algo relajados: los efectos pueden atenuar la barra)
_GREEN_LO = (35, 100, 130)
_GREEN_HI = (80, 255, 255)
_FULL_ORB_S_MIN = 125
_FULL_ORB_V_MIN = 160

_STD_BAR_W = 77
_STD_BAR_H = 7

# Tamaño de captura con la ventana estándar del bot; toda la geometría de este
# módulo está medida a esta escala.
_STD_CAPTURE_W = 550
_STD_CAPTURE_H = 940


def _normalize_scale(screenshot: np.ndarray) -> np.ndarray:
    """Reescala la captura al tamaño estándar si la ventana tiene otro tamaño.

    La proporción de la ventana del juego es fija, así que un resize directo
    no deforma. Permite leer orbes aunque la ventana no se haya podido
    redimensionar (p. ej. el juego corre elevado y SetWindowPos da acceso
    denegado).
    """
    h, w = screenshot.shape[:2]
    if (w, h) == (_STD_CAPTURE_W, _STD_CAPTURE_H):
        return screenshot
    return cv2.resize(screenshot, (_STD_CAPTURE_W, _STD_CAPTURE_H), interpolation=cv2.INTER_AREA)


def _candidate_bars(hsv: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Franjas verdes horizontales candidatas a barra HP, como (x, y, w, h)."""
    green = cv2.inRange(hsv, _GREEN_LO, _GREEN_HI)
    green[:_BAND_Y_MIN, :] = 0
    green[_BAND_Y_MAX:, :] = 0
    green[:, _BAND_X_MAX:] = 0

    n, _, stats, _ = cv2.connectedComponentsWithStats(green)
    pieces = [
        (x, y, w, h)
        for x, y, w, h, _area in stats[1:]
        if w >= 6 and 2 <= h <= 16
    ]
    pieces.sort()

    # Unir fragmentos de una misma barra (números de daño/brillos pueden partirla)
    merged: list[list[int]] = []
    for x, y, w, h in pieces:
        for m in merged:
            if abs(y - m[1]) <= 5 and x - (m[0] + m[2]) <= 40:
                m[2] = max(m[0] + m[2], x + w) - m[0]
                m[3] = max(m[3], h)
                break
        else:
            merged.append([x, y, w, h])

    return [tuple(m) for m in merged if m[2] >= 15]


def _match_formation(cands) -> tuple | None:
    """Ubica las 4 barras de la formación por votación.

    Cada barra candidata, asumida como cada slot de la formación, implica un
    origen (posición del slot 0). Se puntúa cada origen por cuántas candidatas
    caen donde la formación predice; gana el de mayor apoyo (desempate por
    menor desviación acumulada). Las barras sin candidata se reconstruyen.
    """
    if not cands:
        return None

    hypotheses = []  # (support, total_dev, origin, assignment)
    for cand in cands:
        for ox, oy in _FORMATION:
            origin = (cand[0] - ox, cand[1] - oy)

            assignment: list[tuple | None] = []
            support = 0
            total_dev = 0
            for sx, sy in _FORMATION:
                ex, ey = origin[0] + sx, origin[1] + sy
                match, match_dev = None, None
                for c in cands:
                    dx, dy = abs(c[0] - ex), abs(c[1] - ey)
                    if dx <= _TOL_X and dy <= _TOL_Y:
                        dev = dx + dy
                        if match_dev is None or dev < match_dev:
                            match, match_dev = c, dev
                assignment.append(match)
                if match is not None:
                    support += 1
                    total_dev += match_dev

            hypotheses.append((support, total_dev, origin, assignment))

    # Quedarnos con la mejor hipótesis de cada origen (orígenes cercanos = mismo)
    distinct: list[tuple] = []
    for hyp in sorted(hypotheses, key=lambda h: (-h[0], h[1])):
        if any(
            abs(hyp[2][0] - kept[2][0]) <= _TOL_X and abs(hyp[2][1] - kept[2][1]) <= _TOL_Y
            for kept in distinct
        ):
            continue
        distinct.append(hyp)

    best = distinct[0]
    if best[0] < _MIN_BARS_SUPPORT:
        return None

    # Guardia de ambigüedad: otro origen distinto explica igual de bien las barras
    if len(distinct) > 1:
        runner_up = distinct[1]
        if runner_up[0] == best[0] and abs(runner_up[1] - best[1]) <= _AMBIGUITY_DEV_MARGIN:
            return None

    _support, _dev, best_origin, best_assignment = best

    # Reconstruir las barras tapadas con el tamaño típico de las visibles
    seen = [bar for bar in best_assignment if bar is not None]
    std_h = int(np.median([bar[3] for bar in seen])) if seen else _STD_BAR_H

    bars = []
    for slot, bar in enumerate(best_assignment):
        if bar is None:
            sx, sy = _FORMATION[slot]
            bar = (best_origin[0] + sx, best_origin[1] + sy, _STD_BAR_W, std_h)
        bars.append(bar)
    return tuple(bars)


def _count_orbs(hsv: np.ndarray, bar: tuple) -> int | None:
    x, y, w, h = bar
    filled = 0
    for k in range(_ORBS_PER_ALLY):
        cx = int(x + _ORB_DX + k * _ORB_PITCH)
        cy = y + h + _ORB_DY
        cell = hsv[cy : cy + _ORB_SIZE, cx : cx + _ORB_SIZE]
        if cell.size == 0:
            return None
        if cell[..., 1].mean() >= _FULL_ORB_S_MIN and cell[..., 2].mean() >= _FULL_ORB_V_MIN:
            filled += 1
    return filled


# ── Talento de DK ──
# Botón circular con calavera sobre los slots de carta. Cuando el talento está
# ACTIVO el botón entero brilla (el púrpura se va a blanco-rosa); apagado o
# cargando queda oscuro. El discriminador es el BRILLO medio (canal V) de los
# píxeles de la familia púrpura dentro del círculo del botón.
# Calibrado 2026-07-26 con las capturas de images/canopus_dk:
#   dk_talento.png (activo)            -> V medio 247
#   dk_talento_desactivado(.png,_1,_2) -> V medio 129 / 83 / 108
# Umbral 180 deja ~±50 de margen a cada lado.
_TALENT_CENTER = (275, 657)  # a escala 550x940
_TALENT_RADIUS = 20
_TALENT_HUE_LO, _TALENT_HUE_HI = 135, 170  # familia púrpura (H de OpenCV 0-179)
# Si menos de esta fracción del círculo es de la familia púrpura, el botón no está en pantalla.
_TALENT_PRESENCE_MIN = 0.40
_TALENT_READY_V_MEAN = 180


# "Púrpura vivo" (S y V altos): sirve para distinguir las etapas del cooldown.
# Calibrado con las capturas del usuario: dk_talento_desactivado_1 ≈ 4% vivo,
# dk_talento_desactivado_2 ≈ 17%, dk_talento_desactivado (base) ≈ 31%.
_TALENT_VIVID_S_MIN = 120
_TALENT_VIVID_V_MIN = 90


def read_dk_talent_metrics(screenshot: np.ndarray) -> dict:
    """Métricas del botón de talento de DK.

    Returns:
        dict con "estado" ("activo"|"desactivado"|"no_visible"), "brillo"
        (V medio normalizado 0-1 de los píxeles púrpura; ~0.97 activo,
        ~0.3-0.5 desactivado) y "vivid_fraction" (fracción del círculo en
        púrpura vivo; distingue etapas del cooldown).
    """
    screenshot = _normalize_scale(screenshot)
    hsv = cv2.cvtColor(screenshot, cv2.COLOR_BGR2HSV)

    cx, cy = _TALENT_CENTER
    r = _TALENT_RADIUS
    region = hsv[cy - r : cy + r, cx - r : cx + r]
    if region.shape[0] != 2 * r or region.shape[1] != 2 * r:
        return {"estado": "no_visible", "brillo": 0.0, "vivid_fraction": 0.0}

    yy, xx = np.mgrid[-r:r, -r:r]
    mask = (xx * xx + yy * yy) <= r * r

    hue = region[..., 0][mask]
    sat = region[..., 1][mask]
    val = region[..., 2][mask]

    purple_family = (hue >= _TALENT_HUE_LO) & (hue <= _TALENT_HUE_HI)
    if purple_family.sum() / mask.sum() < _TALENT_PRESENCE_MIN:
        return {"estado": "no_visible", "brillo": 0.0, "vivid_fraction": 0.0}

    v_mean = float(val[purple_family].mean())
    vivid = purple_family & (sat >= _TALENT_VIVID_S_MIN) & (val >= _TALENT_VIVID_V_MIN)
    vivid_fraction = float(vivid.sum() / mask.sum())
    estado = "activo" if v_mean >= _TALENT_READY_V_MEAN else "desactivado"
    return {"estado": estado, "brillo": v_mean / 255.0, "vivid_fraction": vivid_fraction}


def read_dk_talent(screenshot: np.ndarray) -> tuple[str, float]:
    """Estado del talento de DK: ("activo"|"desactivado"|"no_visible", brillo 0-1)."""
    metrics = read_dk_talent_metrics(screenshot)
    return metrics["estado"], metrics["brillo"]


def read_ally_orbs(screenshot: np.ndarray) -> list[int] | None:
    """Cuenta los orbes llenos de cada uno de los 3 aliados (DK, Cusack, Galand).

    Tristan (4ta posición, a la derecha) no forma parte de la formación y no
    se detecta.

    Returns:
        Lista de 3 valores 0-5 en orden de pantalla (izquierda a derecha), o
        None si la lectura no es confiable (no se pudieron ubicar suficientes
        barras HP ni por geometría).
    """
    screenshot = _normalize_scale(screenshot)
    hsv = cv2.cvtColor(screenshot, cv2.COLOR_BGR2HSV)
    cands = _candidate_bars(hsv)
    if len(cands) < _MIN_BARS_SUPPORT:
        return None

    bars = _match_formation(cands)
    if bars is None:
        return None

    counts = []
    for bar in bars:
        c = _count_orbs(hsv, bar)
        if c is None:
            return None
        counts.append(c)
    return counts


def debug_annotate(screenshot: np.ndarray) -> tuple[np.ndarray, list, tuple | None, list[int] | None]:
    """Herramienta de calibración: devuelve la imagen anotada y los datos crudos.

    Dibuja en verde las barras candidatas, en amarillo la formación resuelta
    (reconstruidas en rojo) y en cian las celdas de orbes. Devuelve
    (imagen, candidatas, barras_formación, conteos).
    """
    screenshot = _normalize_scale(screenshot)
    hsv = cv2.cvtColor(screenshot, cv2.COLOR_BGR2HSV)
    annotated = screenshot.copy()

    cands = _candidate_bars(hsv)
    for x, y, w, h in cands:
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 1)

    bars = _match_formation(cands)
    counts = None
    if bars is not None:
        counts = []
        for bar in bars:
            x, y, w, h = bar
            color = (0, 255, 255) if bar in cands else (0, 0, 255)
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 1)
            for k in range(_ORBS_PER_ALLY):
                cx = int(x + _ORB_DX + k * _ORB_PITCH)
                cy = y + h + _ORB_DY
                cv2.rectangle(annotated, (cx, cy), (cx + _ORB_SIZE, cy + _ORB_SIZE), (255, 255, 0), 1)
            counts.append(_count_orbs(hsv, bar))

    return annotated, cands, bars, counts


def _main():
    import os
    import sys

    if len(sys.argv) > 1:
        path = sys.argv[1]
        screenshot = cv2.imread(path)
        if screenshot is None:
            print(f"No pude leer la imagen: {path}")
            return
        out_path = f"{os.path.splitext(path)[0]}_orbs_debug.png"
    else:
        from utilities.capture_window import capture_window

        screenshot, _ = capture_window()
        out_path = "canopus_orbs_debug.png"

    annotated, cands, bars, counts = debug_annotate(screenshot)

    print(f"Candidatas a barra HP ({len(cands)}):")
    for bar in cands:
        print(f"  (x={bar[0]}, y={bar[1]}, w={bar[2]}, h={bar[3]})")
    if len(cands) >= 2:
        left = min(cands)
        print(f"Offsets respecto a la candidata más a la izquierda {left[:2]}:")
        for bar in sorted(cands):
            print(f"  (+{bar[0] - left[0]}, {bar[1] - left[1]:+d})")

    if bars is None:
        print("No se pudo resolver la formación (¿offsets de _FORMATION sin calibrar?).")
    else:
        print("Formación resuelta:")
        for slot, bar in enumerate(bars):
            origen = "vista" if bar in cands else "reconstruida"
            nombre = CHARACTER_NAMES[slot]
            print(f"  Slot {slot} ({nombre}): (x={bar[0]}, y={bar[1]}) [{origen}] -> orbes: {counts[slot]}")

    cv2.imwrite(out_path, annotated)
    print(f"Imagen anotada guardada en: {out_path}")


if __name__ == "__main__":
    _main()
